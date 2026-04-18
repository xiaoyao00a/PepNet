import sys
import os
import glob
import time
import json
import random
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import EsmTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score

sys.path.append('../')
from models.Pep2Net_Model import EsmForMultiLabelSequenceClassification, set_seed
from models.FGM import FGM



GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)


# =========================================================================================
#  Augmentation Helpers
# =========================================================================================
BLOSUM_MAPPING = {
    'D': ['E', 'N'], 'E': ['D', 'Q'], 'K': ['R', 'Q'], 'R': ['K'],
    'S': ['T', 'A'], 'T': ['S', 'A'], 'A': ['S', 'G'], 'G': ['A', 'P'],
    'V': ['I', 'L'], 'I': ['V', 'L'], 'L': ['V', 'I'], 'F': ['Y'], 'Y': ['F'],
}

def blosum_substitute(sequence: str, p_sub: float) -> str:
    if p_sub <= 0:
        return sequence
    seq_list = list(sequence)
    for i, aa in enumerate(seq_list):
        if aa in BLOSUM_MAPPING and random.random() < p_sub:
            seq_list[i] = random.choice(BLOSUM_MAPPING[aa])
    return "".join(seq_list)

def span_drop(input_ids: torch.Tensor,
              attention_mask: torch.Tensor,
              mask_token_id: int,
              max_span_len: int,
              p_drop: float) -> torch.Tensor:
    """
    input_ids: [B, L]
    attention_mask: [B, L]
    """
    B, L = input_ids.shape
    output = input_ids.clone()
    for i in range(B):
        if random.random() < p_drop:
            seq_len = int(attention_mask[i].sum().item())
            valid_len = seq_len - 2  
            if valid_len > 0:
                span_len = random.randint(1, min(max_span_len, valid_len))
                start_idx = random.randint(1, seq_len - 1 - span_len)
                end_idx = start_idx + span_len
                output[i, start_idx:end_idx] = mask_token_id
    return output


# =========================================================================================
#  EMA
# =========================================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.decay = decay
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


# =========================================================================================
#  Logger
# =========================================================================================
def init_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


# =========================================================================================
#  Dataset / Collator
# =========================================================================================
class CSVMultiLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_col: str, label_cols):
        self.df = df.reset_index(drop=True).copy()
        self.seq_col = seq_col
        self.label_cols = list(label_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col]).replace(" ", "").upper()
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32), dtype=torch.float32)
        return seq, labels


class Collator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sequences, labels = zip(*batch)
        tokenized = self.tokenizer(
            list(sequences),
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        labels = torch.stack(labels, dim=0)
        return input_ids, attention_mask, labels, list(sequences)


# =========================================================================================
#  Config
# =========================================================================================
class TrainConfig:
    def __init__(self, model_num):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_dir, 'data')

        self.csv_file_name = "peptide.csv"
        self.pretrained_model_dir = "/root/autodl-tmp/esm_small"

        self.data_name = 'Pep2Net_CSV'
        self.model_save_dir = os.path.join(self.project_dir, 'cache', 'Pep2Net')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs', 'Pep2Net')
        self.split_save_dir = os.path.join(self.project_dir, 'cache', 'splits')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.batch_size = 16
        self.max_sen_len = 200
        self.learning_rate = 5e-5
        self.epochs = 100
        self.weight_decay = 0.35
        self.model_num = model_num
        self.seed = GLOBAL_SEED + model_num
        self.num_workers = 0

        self.pad_token_id = 1
        self.mask_token_id = 3
        self.num_labels = None

        self.rdrop_alpha = 0.5
        self.supcon_T = 0.15
        self.supcon_lambda = 0.05
        self.contrastive_dim = 256

        self.p_span_drop = 0.15
        self.max_span_len = 5
        self.p_blosum_sub = 0.01

        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.logs_save_dir, exist_ok=True)
        os.makedirs(self.split_save_dir, exist_ok=True)

        self.log_file = os.path.join(self.logs_save_dir, f'{self.data_name}_run{self.model_num}.log')
        self.best_model_path = os.path.join(self.model_save_dir, f'best_model_{self.data_name}_run{self.model_num}.bin')
        self.best_info_path = os.path.join(self.model_save_dir, f'best_model_{self.data_name}_run{self.model_num}.json')


# =========================================================================================
#  CSV / Split
# =========================================================================================
def find_csv_file(data_dir, csv_file_name=None):
    if csv_file_name is not None:
        exact_path = os.path.join(data_dir, csv_file_name)
        if not os.path.isfile(exact_path):
            raise FileNotFoundError(f"CSV not found: {exact_path}")
        return exact_path

    candidates = sorted(glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No CSV found under: {data_dir}")
    if len(candidates) > 1:
        raise ValueError("Multiple CSV files detected. Please specify the exact file name in config.csv_file_name.\n" + "\n".join(candidates))
    return candidates[0]

def load_csv_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    seq_candidates = [c for c in df.columns if c.lower() == 'sequence']
    if len(seq_candidates) == 0:
        raise ValueError("The CSV must contain a column named 'sequence'.")

    seq_col = seq_candidates[0]
    label_cols = [c for c in df.columns if c != seq_col]

    df = df.dropna(subset=[seq_col]).copy()
    df[seq_col] = df[seq_col].astype(str).str.replace(' ', '', regex=False).str.upper()
    df = df[df[seq_col].str.len() > 0].reset_index(drop=True)

    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).clip(0, 1).astype(np.float32)

    return df, seq_col, label_cols



def _iterative_multilabel_stratified_indices_fallback(label_matrix: np.ndarray,
                                                      test_size: float,
                                                      seed: int):
    Y = np.asarray(label_matrix, dtype=np.int64)
    if Y.ndim != 2:
        raise ValueError("The label_matrix must be a two-dimensional matrix [N, C].")

    n_samples = Y.shape[0]
    if n_samples == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    n_test = int(round(n_samples * test_size))
    n_test = min(max(n_test, 0), n_samples)
    n_train = n_samples - n_test

    if n_train == 0:
        return np.array([], dtype=np.int64), np.arange(n_samples, dtype=np.int64)
    if n_test == 0:
        return np.arange(n_samples, dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.RandomState(seed)

    assigned = np.full(n_samples, fill_value=-1, dtype=np.int64)
    remaining_mask = np.ones(n_samples, dtype=bool)
    remaining_subset_sizes = np.array([n_train, n_test], dtype=np.int64)

    label_totals = Y.sum(axis=0).astype(np.float64)
    desired_label_counts = np.vstack([
        label_totals * (n_train / n_samples),
        label_totals * (n_test / n_samples)
    ])

    row_label_counts = Y.sum(axis=1)

    while remaining_mask.any():
        remaining_indices = np.where(remaining_mask)[0]
        Y_remaining = Y[remaining_indices]
        label_counts_remaining = Y_remaining.sum(axis=0)

        positive_labels = np.where(label_counts_remaining > 0)[0]
        if len(positive_labels) == 0:
            break

        rare_label = positive_labels[np.argmin(label_counts_remaining[positive_labels])]
        candidate_indices = remaining_indices[Y_remaining[:, rare_label] == 1]
        if len(candidate_indices) == 0:
            break

        noise = rng.uniform(0.0, 1e-6, size=len(candidate_indices))
        order = np.argsort(-(row_label_counts[candidate_indices] + noise))
        candidate_indices = candidate_indices[order]

        selected_idx = candidate_indices[0]
        sample_labels = np.where(Y[selected_idx] > 0)[0]

        if len(sample_labels) > 0:
            need_scores = np.maximum(desired_label_counts[:, sample_labels], 0.0).sum(axis=1)
        else:
            need_scores = np.zeros(2, dtype=np.float64)

        if remaining_subset_sizes[0] <= 0:
            chosen_subset = 1
        elif remaining_subset_sizes[1] <= 0:
            chosen_subset = 0
        else:
            max_need = need_scores.max()
            best_subsets = np.where(np.isclose(need_scores, max_need))[0]

            if len(best_subsets) > 1:
                subset_caps = remaining_subset_sizes[best_subsets]
                best_subsets = best_subsets[subset_caps == subset_caps.max()]

            chosen_subset = int(rng.choice(best_subsets))

        assigned[selected_idx] = chosen_subset
        remaining_mask[selected_idx] = False
        remaining_subset_sizes[chosen_subset] -= 1

        if len(sample_labels) > 0:
            desired_label_counts[chosen_subset, sample_labels] -= 1.0

    leftovers = np.where(remaining_mask)[0]
    rng.shuffle(leftovers)

    for idx in leftovers:
        if remaining_subset_sizes[0] <= 0:
            chosen_subset = 1
        elif remaining_subset_sizes[1] <= 0:
            chosen_subset = 0
        else:
            best_subsets = np.where(remaining_subset_sizes == remaining_subset_sizes.max())[0]
            chosen_subset = int(rng.choice(best_subsets))

        assigned[idx] = chosen_subset
        remaining_subset_sizes[chosen_subset] -= 1

        sample_labels = np.where(Y[idx] > 0)[0]
        if len(sample_labels) > 0:
            desired_label_counts[chosen_subset, sample_labels] -= 1.0

    train_idx = np.where(assigned == 0)[0]
    test_idx = np.where(assigned == 1)[0]

    return train_idx, test_idx


def multilabel_stratified_train_test_split_indices(label_matrix: np.ndarray,
                                                   test_size: float = 0.2,
                                                   seed: int = 42):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        X_dummy = np.zeros((len(label_matrix), 1), dtype=np.float32)
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed
        )
        train_idx, test_idx = next(splitter.split(X_dummy, label_matrix))
        logging.info("Using MultilabelStratifiedShuffleSplit from iterstrat.")
        return train_idx, test_idx

    except Exception as e:
        logging.info(f" {repr(e)}")
        return _iterative_multilabel_stratified_indices_fallback(
            label_matrix=label_matrix,
            test_size=test_size,
            seed=seed
        )


def split_dataframe_4_to_1_multilabel_stratified(df, label_cols, seed, test_size=0.2):
    label_matrix = df[label_cols].values.astype(np.int64)
    train_idx, test_idx = multilabel_stratified_train_test_split_indices(
        label_matrix=label_matrix,
        test_size=test_size,
        seed=seed
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def log_split_label_distribution(train_df, test_df, label_cols):
    train_counts = train_df[label_cols].sum(axis=0).values.astype(np.int64)
    test_counts = test_df[label_cols].sum(axis=0).values.astype(np.int64)
    total_counts = train_counts + test_counts

    logging.info("Label distribution after multilabel stratified split:")
    logging.info("Idx\tLabel\tTrainPos\tTestPos\tTotalPos\tTestRatio")
    for i, (name, tr, te, tt) in enumerate(zip(label_cols, train_counts, test_counts, total_counts)):
        ratio = (te / tt) if tt > 0 else 0.0
        logging.info(f"{i:02d}\t{name}\t{int(tr)}\t{int(te)}\t{int(tt)}\t{ratio:.3f}")


def build_dataloaders(df, seq_col, label_cols, tokenizer, config):
    train_df, test_df = split_dataframe_4_to_1_multilabel_stratified(
        df=df,
        label_cols=label_cols,
        seed=config.seed,
        test_size=0.2
    )

    train_df.to_csv(os.path.join(config.split_save_dir, f'train_run{config.model_num}.csv'), index=False)
    test_df.to_csv(os.path.join(config.split_save_dir, f'test_run{config.model_num}.csv'), index=False)

    log_split_label_distribution(train_df, test_df, label_cols)

    train_dataset = CSVMultiLabelDataset(train_df, seq_col, label_cols)
    test_dataset = CSVMultiLabelDataset(test_df, seq_col, label_cols)
    collator = Collator(tokenizer, max_length=config.max_sen_len)

    g = torch.Generator()
    g.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator
    )

    return train_loader, test_loader, train_df, test_df


# =========================================================================================
#  Param Groups
# =========================================================================================
def build_param_groups(model, base_lr, wd):
    all_params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    def pop_by_prefix(prefixes):
        picked, to_delete = [], []
        for n, p in list(all_params.items()):
            if any(n == pref or n.startswith(pref + '.') for pref in prefixes):
                picked.append(p)
                to_delete.append(n)
        for n in to_delete:
            all_params.pop(n)
        return picked

    esm_params = pop_by_prefix(['esm'])
    fusion_params = pop_by_prefix(['fusion_block', 'physico_branch'])
    head_params = pop_by_prefix(['msd_head'])
    contrastive_params = pop_by_prefix(['proj', 'prototypes'])
    others_params = list(all_params.values())

    param_groups = []
    if esm_params:
        param_groups.append({'params': esm_params, 'lr': base_lr, 'weight_decay': wd})
    if fusion_params:
        param_groups.append({'params': fusion_params, 'lr': base_lr * 5, 'weight_decay': 0.0})
    if head_params:
        param_groups.append({'params': head_params, 'lr': base_lr * 10, 'weight_decay': wd})
    if contrastive_params:
        param_groups.append({'params': contrastive_params, 'lr': base_lr * 10, 'weight_decay': wd})
    if others_params:
        param_groups.append({'params': others_params, 'lr': base_lr * 5, 'weight_decay': wd})

    return param_groups


# =========================================================================================
#  Loss / Metrics
# =========================================================================================
def symmetric_bernoulli_kl(p, q, eps=1e-7):
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)

    kl_pq = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
    kl_qp = q * (q.log() - p.log()) + (1 - q) * ((1 - q).log() - (1 - p).log())

    return (kl_pq + kl_qp).mean()

def supcon_multilabel(z, labels, T=0.07):
    B = z.size(0)
    sim = torch.matmul(z, z.t()) / T
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

    with torch.no_grad():
        pos_mask = (labels @ labels.t()) > 0
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask = pos_mask & (~eye)
        denom_mask = ~eye
        valid = pos_mask.sum(dim=1) > 0
        if not valid.any():
            return torch.tensor(0.0, device=z.device)

    exp_sim = torch.exp(sim) * denom_mask
    denom = exp_sim.sum(dim=1).clamp(min=1e-8)
    pos_sum = (exp_sim * pos_mask).sum(dim=1).clamp(min=1e-8)
    loss_vec = -torch.log(pos_sum / denom)
    return loss_vec[valid].mean()

def calculate_class_balanced_weights_from_matrix(label_matrix, beta=0.999):
    class_counts = torch.tensor(label_matrix.sum(axis=0), dtype=torch.float32)
    effective_num = 1.0 - torch.pow(beta, class_counts)
    effective_num = torch.where(class_counts > 0, effective_num, torch.ones_like(effective_num))

    weights = (1.0 - beta) / effective_num
    weights = torch.where(class_counts > 0, weights, torch.zeros_like(weights))

    if weights.sum() > 0:
        weights = weights / weights.sum() * len(weights)
    else:
        weights = torch.ones_like(weights)

    return weights

def _safe_divide(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)

def _fmt_metric(x):
    return f"{x:.4f}" if np.isfinite(x) else "nan"

def evaluate_multilabel(pred_bin: np.ndarray, true_bin: np.ndarray):

    pred_bin = pred_bin.astype(np.int64)
    true_bin = true_bin.astype(np.int64)

    tp = ((pred_bin == 1) & (true_bin == 1)).sum(axis=1).astype(np.float64)
    fp = ((pred_bin == 1) & (true_bin == 0)).sum(axis=1).astype(np.float64)
    fn = ((pred_bin == 0) & (true_bin == 1)).sum(axis=1).astype(np.float64)

    aiming = _safe_divide(tp, tp + fp).mean()
    coverage = _safe_divide(tp, tp + fn).mean()
    accuracy = _safe_divide(tp, tp + fp + fn).mean()
    absolute_true = np.mean(np.all(pred_bin == true_bin, axis=1).astype(np.float64))
    absolute_false = np.mean((fp + fn) / true_bin.shape[1])

    return aiming, coverage, accuracy, absolute_true, absolute_false

def compute_multilabel_paper_metrics(pred_prob: np.ndarray, true_bin: np.ndarray, threshold=0.5):

    pred_prob = np.asarray(pred_prob, dtype=np.float64)
    true_bin = np.asarray(true_bin, dtype=np.int64)

    if pred_prob.shape != true_bin.shape:
        raise ValueError(f"Shape mismatch: pred_prob={pred_prob.shape}, true_bin={true_bin.shape}")

    if true_bin.size == 0:
        return {
            'pred_bin': np.zeros_like(true_bin, dtype=np.int64),
            'aiming': 0.0,
            'coverage': 0.0,
            'accuracy': 0.0,
            'absolute_true': 0.0,
            'absolute_false': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'f1': 0.0,
            'acc': 0.0,
            'mcc': 0.0,
            'auc_macro': 0.0,
            'auc_micro': 0.0,
        }

    pred_bin = (pred_prob >= threshold).astype(np.int64)

    tp = ((pred_bin == 1) & (true_bin == 1)).sum(axis=1).astype(np.float64)
    fp = ((pred_bin == 1) & (true_bin == 0)).sum(axis=1).astype(np.float64)
    fn = ((pred_bin == 0) & (true_bin == 1)).sum(axis=1).astype(np.float64)
    tn = ((pred_bin == 0) & (true_bin == 0)).sum(axis=1).astype(np.float64)

    precision = _safe_divide(tp, tp + fp).mean()
    recall = _safe_divide(tp, tp + fn).mean()
    specificity = _safe_divide(tn, tn + fp).mean()
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn).mean()
    acc = _safe_divide(tp, tp + fp + fn).mean()

    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_divide(tp * tn - fp * fn, mcc_den).mean()

    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate_multilabel(pred_bin, true_bin)

    auc_list = []
    for c in range(true_bin.shape[1]):
        yc = true_bin[:, c]
        pc = pred_prob[:, c]
        if np.unique(yc).size < 2:
            continue
        try:
            auc_list.append(roc_auc_score(yc, pc))
        except Exception:
            continue
    auc_macro = float(np.mean(auc_list)) if len(auc_list) > 0 else np.nan

    flat_true = true_bin.reshape(-1)
    flat_prob = pred_prob.reshape(-1)
    if np.unique(flat_true).size >= 2:
        try:
            auc_micro = float(roc_auc_score(flat_true, flat_prob))
        except Exception:
            auc_micro = np.nan
    else:
        auc_micro = np.nan

    return {
        'pred_bin': pred_bin,
        'aiming': float(aiming),
        'coverage': float(coverage),
        'accuracy': float(accuracy),
        'absolute_true': float(absolute_true),
        'absolute_false': float(absolute_false),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'acc': float(acc),
        'mcc': float(mcc),
        'auc_macro': float(auc_macro) if np.isfinite(auc_macro) else np.nan,
        'auc_micro': float(auc_micro) if np.isfinite(auc_micro) else np.nan,
    }

def compute_binary_metrics_one_class(y_true: np.ndarray, y_prob: np.ndarray, threshold=0.5):

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / mcc_den) if mcc_den > 0 else 0.0

    if np.unique(y_true).size >= 2:
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except Exception:
            auc_score = np.nan
    else:
        auc_score = np.nan

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'positives': int((y_true == 1).sum()),
        'negatives': int((y_true == 0).sum()),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'auc': float(auc_score) if np.isfinite(auc_score) else np.nan,
        'acc': float(acc),
        'mcc': float(mcc),
    }

def compute_per_class_binary_metrics(pred_prob: np.ndarray,
                                     true_bin: np.ndarray,
                                     label_cols,
                                     threshold=0.5):

    pred_prob = np.asarray(pred_prob, dtype=np.float64)
    true_bin = np.asarray(true_bin, dtype=np.int64)

    results = []
    num_labels = true_bin.shape[1]

    for i in range(num_labels):
        one = compute_binary_metrics_one_class(
            y_true=true_bin[:, i],
            y_prob=pred_prob[:, i],
            threshold=threshold
        )
        one['idx'] = i
        one['label'] = label_cols[i] if label_cols is not None else f"class_{i}"
        results.append(one)

    return results

def summarize_per_class_metrics(per_class_metrics):
    if len(per_class_metrics) == 0:
        return {
            'recall': 0.0,
            'specificity': 0.0,
            'f1': 0.0,
            'auc': np.nan,
            'acc': 0.0,
            'mcc': 0.0,
        }

    recall = np.mean([x['recall'] for x in per_class_metrics])
    specificity = np.mean([x['specificity'] for x in per_class_metrics])
    f1 = np.mean([x['f1'] for x in per_class_metrics])
    acc = np.mean([x['acc'] for x in per_class_metrics])
    mcc = np.mean([x['mcc'] for x in per_class_metrics])

    auc_vals = [x['auc'] for x in per_class_metrics if np.isfinite(x['auc'])]
    auc_score = np.mean(auc_vals) if len(auc_vals) > 0 else np.nan

    return {
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'auc': float(auc_score) if np.isfinite(auc_score) else np.nan,
        'acc': float(acc),
        'mcc': float(mcc),
    }

def log_per_class_metrics_table(title: str, per_class_metrics):
    logging.info("\n" + title)
    logging.info("Idx\tLabel\tPos\tNeg\tRecall\tSpecificity\tF1-score\tAUC\tACC\tMCC")
    for item in per_class_metrics:
        logging.info(
            f"{item['idx']:02d}\t"
            f"{item['label']}\t"
            f"{item['positives']}\t"
            f"{item['negatives']}\t"
            f"{_fmt_metric(item['recall'])}\t"
            f"{_fmt_metric(item['specificity'])}\t"
            f"{_fmt_metric(item['f1'])}\t"
            f"{_fmt_metric(item['auc'])}\t"
            f"{_fmt_metric(item['acc'])}\t"
            f"{_fmt_metric(item['mcc'])}"
        )


# =========================================================================================
#  Test
# =========================================================================================
def test_model(data_loader, model, config, label_cols=None, threshold=0.5, return_per_class=False):
    model.eval()
    with torch.no_grad():
        pred_res, real_res = [], []

        for input_ids, attention_mask, labels, _ in data_loader:
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)

            pred_res.append(torch.sigmoid(logits).cpu().numpy())
            real_res.append(labels.cpu().numpy())

        pred_res = np.vstack(pred_res) if len(pred_res) > 0 else np.zeros((0, config.num_labels), dtype=np.float32)
        real_res = np.vstack(real_res) if len(real_res) > 0 else np.zeros((0, config.num_labels), dtype=np.float32)
        real_bin = real_res.astype(np.int64)

        metrics = compute_multilabel_paper_metrics(pred_res, real_bin, threshold=threshold)

        if return_per_class:
            per_class_metrics = compute_per_class_binary_metrics(
                pred_prob=pred_res,
                true_bin=real_bin,
                label_cols=label_cols,
                threshold=threshold
            )
            per_class_macro = summarize_per_class_metrics(per_class_metrics)

            metrics['per_class_metrics'] = per_class_metrics
            metrics['per_class_macro'] = per_class_macro

    model.train()
    return metrics


# =========================================================================================
#  Training
# =========================================================================================
def train(config):
    set_seed(config.seed)
    init_logger(config.log_file)

    logging.info(f"Run {config.model_num} started, seed={config.seed}, device={config.device}")

    # 1)  CSV
    csv_path = find_csv_file(config.data_dir, config.csv_file_name)
    df, seq_col, label_cols = load_csv_dataframe(csv_path)

    config.num_labels = len(label_cols)
    config.data_name = os.path.splitext(os.path.basename(csv_path))[0]

    logging.info(f"CSV file: {csv_path}")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Sequence column: {seq_col}")
    logging.info(f"Num labels: {config.num_labels}")
    logging.info(f"Label columns: {label_cols}")

    # 2) tokenizer
    tokenizer = EsmTokenizer.from_pretrained(config.pretrained_model_dir)
    config.pad_token_id = tokenizer.pad_token_id
    config.mask_token_id = tokenizer.mask_token_id

    # 3) dataloader
    train_loader, test_loader, train_df, test_df = build_dataloaders(df, seq_col, label_cols, tokenizer, config)

    logging.info(f"Train size: {len(train_df)}")
    logging.info(f"Test size: {len(test_df)}")

    # 4) model
    model = EsmForMultiLabelSequenceClassification(
        esm_pretrained_model_dir=config.pretrained_model_dir,
        num_labels=config.num_labels,
        fusion_dim=256,
        contrastive_dim=config.contrastive_dim,
        tokenizer=tokenizer
    ).to(config.device)

    ema = EMA(model, decay=0.999)
    fgm = FGM(model)

    optimizer_grouped_parameters = build_param_groups(model, base_lr=config.learning_rate, wd=config.weight_decay)
    optimizer = AdamW(optimizer_grouped_parameters)

    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.06)),
        num_training_steps=total_steps
    )

    cb_weights = calculate_class_balanced_weights_from_matrix(train_df[label_cols].values).to(config.device)
    logging.info(f"Class Balanced Weights: {cb_weights.detach().cpu().numpy()}")

    best_score = -1.0
    best_epoch = -1

    logging.info(
        f"[Config] rdrop_alpha={config.rdrop_alpha}, "
        f"supcon_lambda={config.supcon_lambda}, "
        f"p_span_drop={config.p_span_drop}, "
        f"p_blosum_sub={config.p_blosum_sub}"
    )

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for idx, (input_ids, attention_mask, labels, sequences) in enumerate(train_loader):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            # ---------- Augmentation 1: BLOSUM ----------
            sequences_aug1 = [blosum_substitute(s, config.p_blosum_sub) for s in sequences]
            tokenized_aug1 = tokenizer(
                sequences_aug1,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=config.max_sen_len
            )
            sample_aug1 = tokenized_aug1['input_ids'].to(config.device)
            attention_mask_aug1 = tokenized_aug1['attention_mask'].to(config.device)

            # ---------- Augmentation 2: Span Drop ----------
            sample_aug2 = span_drop(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_token_id=config.mask_token_id,
                max_span_len=config.max_span_len,
                p_drop=config.p_span_drop
            )
            attention_mask_aug2 = attention_mask

            optimizer.zero_grad()

            # ---------- R-Drop ----------
            loss1, logits1, z1, _ = model(
                input_ids=sample_aug1,
                attention_mask=attention_mask_aug1,
                labels=labels,
                cb_weights=cb_weights
            )
            loss2, logits2, z2, _ = model(
                input_ids=sample_aug2,
                attention_mask=attention_mask_aug2,
                labels=labels,
                cb_weights=cb_weights
            )

            loss_sup = 0.5 * (loss1 + loss2)
            p1, p2 = torch.sigmoid(logits1), torch.sigmoid(logits2)
            loss_rdrop = config.rdrop_alpha * symmetric_bernoulli_kl(p1, p2)

            z_supcon = torch.stack([z1, z2], dim=1).reshape(-1, z1.size(-1))
            labels_supcon = labels.unsqueeze(1).expand(-1, 2, -1).reshape(-1, labels.size(-1))
            loss_supcon = supcon_multilabel(z_supcon, labels_supcon, T=config.supcon_T) * config.supcon_lambda

            loss = loss_sup + loss_rdrop + loss_supcon
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ---------- FGM ----------
            fgm.attack()
            loss_adv, _, _, _ = model(
                input_ids=sample_aug2,
                attention_mask=attention_mask_aug2,
                labels=labels,
                cb_weights=cb_weights
            )
            loss_adv.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            fgm.restore()

            optimizer.step()
            scheduler.step()
            ema.update(model)

            running_loss += loss.item()

            if idx % 50 == 0:
                logging.info(
                    f"Epoch [{epoch + 1}/{config.epochs}] "
                    f"Batch [{idx}/{len(train_loader)}] "
                    f"LR={scheduler.get_last_lr()[0]:.6e} "
                    f"Loss={loss.item():.4f} "
                    f"(sup={loss_sup.item():.4f}, rdrop={loss_rdrop.item():.4f}, supcon={loss_supcon.item():.4f})"
                )

 
        ema.apply_to(model)

        train_metrics = test_model(train_loader, model, config, label_cols=label_cols, return_per_class=True)
        test_metrics = test_model(test_loader, model, config, label_cols=label_cols, return_per_class=True)

        avg_loss = running_loss / max(1, len(train_loader))
        epoch_time = time.time() - start_time

        test_per_class_metrics = test_metrics['per_class_metrics']
        test_per_class_macro = test_metrics['per_class_macro']

        logging.info(
            f"Epoch {epoch + 1} finished | "
            f"time={epoch_time:.2f}s | avg_loss={avg_loss:.4f} | "
            f"train_aiming={train_metrics['aiming']:.4f} | "
            f"test_aiming={test_metrics['aiming']:.4f} | "
            f"test_coverage={test_metrics['coverage']:.4f} | "
            f"test_accuracy={test_metrics['accuracy']:.4f} | "
            f"test_abs_true={test_metrics['absolute_true']:.4f} | "
            f"test_abs_false={test_metrics['absolute_false']:.4f} | "
            f"test_recall={test_metrics['recall']:.4f} | "
            f"test_specificity={test_metrics['specificity']:.4f} | "
            f"test_f1={test_metrics['f1']:.4f} | "
            f"test_auc_macro={_fmt_metric(test_metrics['auc_macro'])} | "
            f"test_auc_micro={_fmt_metric(test_metrics['auc_micro'])} | "
            f"test_acc={test_metrics['acc']:.4f} | "
            f"test_mcc={test_metrics['mcc']:.4f}"
        )

        logging.info(
            f"[Epoch {epoch + 1}] TEST per-class macro summary | "
            f"Recall={test_per_class_macro['recall']:.4f} | "
            f"Specificity={test_per_class_macro['specificity']:.4f} | "
            f"F1-score={test_per_class_macro['f1']:.4f} | "
            f"AUC={_fmt_metric(test_per_class_macro['auc'])} | "
            f"ACC={test_per_class_macro['acc']:.4f} | "
            f"MCC={test_per_class_macro['mcc']:.4f}"
        )

        print(
            f"Epoch {epoch + 1}: "
            f"Recall={test_metrics['recall']:.4f}, "
            f"Specificity={test_metrics['specificity']:.4f}, "
            f"F1-score={test_metrics['f1']:.4f}, "
            f"AUC_macro={_fmt_metric(test_metrics['auc_macro'])}, "
            f"ACC={test_metrics['acc']:.4f}, "
            f"MCC={test_metrics['mcc']:.4f}"
        )


        log_per_class_metrics_table(
            f"[Epoch {epoch + 1}] TEST per-class metrics",
            test_per_class_metrics
        )


        test_score = test_metrics['aiming'] + test_metrics['coverage'] + test_metrics['accuracy']

        if test_score > best_score:
            best_score = test_score
            best_epoch = epoch + 1

            torch.save(model.state_dict(), config.best_model_path)

            best_info = {
                "csv_path": csv_path,
                "run_id": config.model_num,
                "seed": config.seed,
                "best_epoch": best_epoch,
                "best_test_score": float(best_score),

                "best_test_aiming": float(test_metrics['aiming']),
                "best_test_coverage": float(test_metrics['coverage']),
                "best_test_accuracy": float(test_metrics['accuracy']),
                "best_test_absolute_true": float(test_metrics['absolute_true']),
                "best_test_absolute_false": float(test_metrics['absolute_false']),

                "best_test_precision": float(test_metrics['precision']),
                "best_test_recall": float(test_metrics['recall']),
                "best_test_specificity": float(test_metrics['specificity']),
                "best_test_f1": float(test_metrics['f1']),
                "best_test_acc": float(test_metrics['acc']),
                "best_test_mcc": float(test_metrics['mcc']),
                "best_test_auc_macro": None if not np.isfinite(test_metrics['auc_macro']) else float(test_metrics['auc_macro']),
                "best_test_auc_micro": None if not np.isfinite(test_metrics['auc_micro']) else float(test_metrics['auc_micro']),

                "best_test_per_class_macro_recall": float(test_per_class_macro['recall']),
                "best_test_per_class_macro_specificity": float(test_per_class_macro['specificity']),
                "best_test_per_class_macro_f1": float(test_per_class_macro['f1']),
                "best_test_per_class_macro_auc": None if not np.isfinite(test_per_class_macro['auc']) else float(test_per_class_macro['auc']),
                "best_test_per_class_macro_acc": float(test_per_class_macro['acc']),
                "best_test_per_class_macro_mcc": float(test_per_class_macro['mcc']),

                "num_labels": config.num_labels,
                "label_cols": label_cols,
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
            }

            with open(config.best_info_path, 'w', encoding='utf-8') as f:
                json.dump(best_info, f, ensure_ascii=False, indent=2)

            logging.info(
                f"[Best Model Updated] epoch={best_epoch}, "
                f"test_score={best_score:.4f} | "
                f"Recall={test_metrics['recall']:.4f}, "
                f"Specificity={test_metrics['specificity']:.4f}, "
                f"F1={test_metrics['f1']:.4f}, "
                f"AUC_macro={_fmt_metric(test_metrics['auc_macro'])}, "
                f"ACC={test_metrics['acc']:.4f}, "
                f"MCC={test_metrics['mcc']:.4f} | "
                f"saved to {config.best_model_path}"
            )

    
        ema.restore(model)

    logging.info(
        f"Run {config.model_num} finished. "
        f"Best epoch={best_epoch}, best test score={best_score:.4f}"
    )



if __name__ == '__main__':
    NUM_RUNS = 10  

    for model_num in range(NUM_RUNS):
        print("\n" + "=" * 80)
        print(f"Starting run={model_num}, seed={GLOBAL_SEED + model_num}")
        print("=" * 80)

        cfg = TrainConfig(model_num=model_num)
        train(cfg)