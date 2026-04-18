import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import os
from transformers import EsmModel


def set_seed(seed=42):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================================
#  Physicochemical Feature Definitions (Fixed, Not Learned)
# =========================================================================================

# 20 standard amino acids
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# ---------- PC6 Encoding (6 physicochemical properties per residue) ----------
PC6_TABLE = {
    'A': [ 0.008,  0.134, -0.475, -0.039,  0.181, -0.143],
    'C': [ 0.248,  0.134, -0.578, -0.024, -0.104, -0.017],
    'D': [-0.236, -0.285,  0.582, -0.030, -0.157,  0.260],
    'E': [-0.222, -0.018,  0.475,  0.003, -0.137,  0.321],
    'F': [ 0.438,  0.489, -0.568,  0.050,  0.085, -0.096],
    'G': [-0.073,  0.134, -0.378, -0.039,  0.065, -0.190],
    'H': [ 0.020,  0.134,  0.064,  0.061, -0.010,  0.131],
    'I': [ 0.449,  0.287, -0.475, -0.039,  0.185, -0.143],
    'K': [-0.325, -0.018,  0.514,  0.073, -0.137,  0.321],
    'L': [ 0.449,  0.244, -0.475, -0.039,  0.185, -0.143],
    'M': [ 0.295,  0.244, -0.475,  0.022,  0.140, -0.096],
    'N': [-0.150, -0.134,  0.390, -0.018, -0.104,  0.201],
    'P': [-0.168,  0.134, -0.173, -0.039, -0.340, -0.096],
    'Q': [-0.198, -0.018,  0.390,  0.003, -0.104,  0.260],
    'R': [-0.274, -0.018,  0.582,  0.109, -0.137,  0.321],
    'S': [-0.073, -0.134,  0.064, -0.039, -0.015,  0.060],
    'T': [-0.073,  0.134,  0.064, -0.039,  0.015,  0.060],
    'V': [ 0.339,  0.244, -0.475, -0.039,  0.185, -0.143],
    'W': [ 0.379,  0.690, -0.568,  0.050,  0.065, -0.096],
    'Y': [ 0.260,  0.489, -0.173,  0.050,  0.015, -0.017],
}

# ---------- AAindex: 12 Selected Physicochemical Indices ----------
AAINDEX_TABLE = {
    #        Hydro   Polar  VdWVol  Charge   pI    SolvEn  Eisenb  Flex   Helix  Sheet  MW     ASA
    'A': [ 1.800,  8.100,  0.167,  0.000,  6.00, -0.368,  0.620,  0.360, 1.420, 0.830, 0.491, 1.181],
    'R': [-4.500, 10.500,  0.291,  1.000, 10.76, -1.030, -2.530,  0.530, 0.980, 0.930, 1.204, 2.560],
    'N': [-3.500, 11.600,  0.212,  0.000,  5.41, -0.910, -0.780,  0.460, 0.670, 0.890, 0.912, 1.655],
    'D': [-3.500, 13.000,  0.187,  -1.00,  2.77, -0.600, -0.900,  0.510, 1.010, 0.540, 0.921, 1.587],
    'C': [ 2.500,  5.500,  0.201,  0.000,  5.07, -0.205,  0.290,  0.350, 0.700, 1.190, 0.839, 1.461],
    'E': [-3.500, 12.300,  0.223,  -1.00,  3.22, -0.650, -0.740,  0.500, 1.510, 0.370, 1.015, 1.862],
    'Q': [-3.500, 10.500,  0.259,  0.000,  5.65, -0.850, -0.850,  0.490, 1.110, 1.100, 1.008, 1.932],
    'G': [-0.400,  9.000,  0.000,  0.000,  5.97, -0.525,  0.480,  0.540, 0.570, 0.750, 0.397, 0.881],
    'H': [-3.200, 10.400,  0.242,  0.500,  7.59, -0.400, -0.400,  0.320, 1.000, 0.870, 1.074, 2.025],
    'I': [ 4.500,  5.200,  0.293,  0.000,  6.02, -0.230,  1.380,  0.460, 1.080, 1.600, 0.906, 1.810],
    'L': [ 3.800,  4.900,  0.293,  0.000,  5.98, -0.193,  1.060,  0.370, 1.210, 1.300, 0.906, 1.931],
    'K': [-3.900, 11.300,  0.283,  1.000,  9.74, -1.050, -1.500,  0.470, 1.160, 0.740, 1.007, 2.258],
    'M': [ 1.900,  5.700,  0.275,  0.000,  5.74, -0.282,  0.640,  0.300, 1.450, 1.050, 1.030, 2.034],
    'F': [ 2.800,  5.200,  0.353,  0.000,  5.48, -0.155,  1.190,  0.310, 1.130, 1.380, 1.142, 2.228],
    'P': [-1.600,  8.000,  0.195,  0.000,  6.30, -0.294, -0.120,  0.510, 0.570, 0.550, 0.793, 1.468],
    'S': [-0.800,  9.200,  0.132,  0.000,  5.68, -0.524,  0.180,  0.510, 0.770, 0.750, 0.726, 1.298],
    'T': [-0.700,  8.600,  0.195,  0.000,  5.60, -0.400,  0.050,  0.440, 0.830, 1.190, 0.824, 1.525],
    'W': [-0.900,  5.400,  0.432,  0.000,  5.89, -0.050,  0.810,  0.310, 1.080, 1.370, 1.411, 2.663],
    'Y': [-1.300,  6.200,  0.370,  0.000,  5.66, -0.110, -0.260,  0.420, 0.690, 1.470, 1.255, 2.368],
    'V': [ 4.200,  5.900,  0.234,  0.000,  5.96, -0.198,  1.080,  0.390, 1.060, 1.700, 0.811, 1.645],
}

# ---------- BLOSUM62 Substitution Matrix (20x20 profile per AA) ----------
BLOSUM62_MATRIX = {
    'A': [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-1,-1, 1, 0,-3,-2, 0],
    'C': [-1, 9,-3,-4,-2,-3,-4,-3,-3,-1,-1,-3,-1,-3,-3,-1,-1,-2,-2,-1],
    'D': [-2,-3, 6, 2,-3,-1, 2,-1,-1,-3,-4,-1,-1, 0,-2, 0,-1,-4,-3,-3],
    'E': [-2,-4, 2, 5,-3,-2, 0,-2, 0,-3,-3, 1,-1, 2, 0, 0,-1,-3,-2,-2],
    'F': [ 0,-2,-3,-3, 6,-3,-3,-3,-1, 0, 0,-3,-4,-3,-3,-2,-2, 1, 3,-1],
    'G': [-1,-3,-1,-2,-3, 6,-2,-1,-2,-4,-4,-2,-2,-2,-2, 0,-2,-2,-3,-3],
    'H': [-1,-4, 2, 0,-3,-2, 8,-2,-1,-3,-3,-1,-2, 0, 0,-1,-2,-2, 2,-3],
    'I': [ 0,-3,-1,-2,-3,-1,-2, 4,-3, 0,-1,-2,-3,-2,-1, 0,-1,-3,-3, 3],
    'K': [-2,-3,-1, 0,-1,-2,-1,-3, 5,-2,-3, 4,-1, 1, 2, 0,-1,-3,-2,-2],
    'L': [-1,-1,-3,-3, 0,-4,-3, 0,-2, 4, 2,-3,-3,-2,-2,-2,-1,-2,-1, 1],
    'M': [-1,-1,-4,-3, 0,-4,-3,-1,-3, 2, 5,-2,-2, 0,-1,-1,-1,-1,-1, 1],
    'N': [-1,-3,-1, 1,-3,-2,-1,-2, 4,-3,-2, 5,-2, 0, 0, 1, 0,-4,-2,-3],
    'P': [-1,-1,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-4,-3,-2],
    'Q': [-1,-3, 0, 2,-3,-2, 0,-2, 1,-2, 0, 0,-1, 5, 1, 0,-1,-2,-1,-2],
    'R': [-1,-3,-2, 0,-3,-2, 0,-1, 2,-2,-1, 0,-2, 1, 5,-1,-1,-3,-2,-3],
    'S': [ 1,-1, 0, 0,-2, 0,-1, 0, 0,-2,-1, 1,-1, 0,-1, 4, 1,-3,-2,-2],
    'T': [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5,-2,-2, 0],
    'V': [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0, 4,-3,-1],
    'W': [-3,-2,-4,-3, 1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11, 2],
    'Y': [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1, 2, 7],
}

PC6_DIM = 6
AAINDEX_DIM = 12
BLOSUM62_DIM = 20
RESIDUE_FEATURE_DIM = PC6_DIM + AAINDEX_DIM + BLOSUM62_DIM  # 6 + 12 + 20 = 38

# Global composition feature dimensions
AAC_DIM = 20       # Amino Acid Composition
DPC_DIM = 400      # Dipeptide Composition
PAAC_DIM = 50      # Pseudo Amino Acid Composition (20 AAC + 30 sequence-order)
GLOBAL_FEATURE_DIM = AAC_DIM + DPC_DIM + PAAC_DIM  # 20 + 400 + 50 = 470



def _build_residue_feature_table(tokenizer):
    """
    Build a fixed lookup table mapping ESM token IDs to physicochemical feature vectors.
    Returns a tensor of shape [vocab_size, RESIDUE_FEATURE_DIM].
    Non-standard / special tokens get zero vectors.

    Changed from original: each feature group (PC6, AAindex, BLOSUM62) is Z-score
    normalized independently before concatenation, so no single group dominates
    due to scale differences.
    """
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 33

    # Collect features per group
    aa_ids, pc6_list, aaindex_list, blosum_list = [], [], [], []
    for aa in STANDARD_AAS:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id is None or token_id == tokenizer.unk_token_id:
            continue
        aa_ids.append(token_id)
        pc6_list.append(PC6_TABLE[aa])
        aaindex_list.append(AAINDEX_TABLE[aa])
        blosum_list.append(BLOSUM62_MATRIX[aa])

    def group_normalize(feat_list):
        arr = np.array(feat_list, dtype=np.float32)
        mean = arr.mean(axis=0, keepdims=True)
        std  = arr.std(axis=0, keepdims=True)
        std  = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std

    pc6_norm     = group_normalize(pc6_list)     # [20, 6]
    aaindex_norm = group_normalize(aaindex_list) # [20, 12]
    blosum_norm  = group_normalize(blosum_list)  # [20, 20]

    table = np.zeros((vocab_size, RESIDUE_FEATURE_DIM), dtype=np.float32)
    for i, token_id in enumerate(aa_ids):
        table[token_id] = np.concatenate([pc6_norm[i], aaindex_norm[i], blosum_norm[i]])

    return torch.from_numpy(table)


def _token_ids_to_aa_sequences(input_ids, tokenizer):
    """
    Convert token IDs back to amino acid character sequences for global feature computation.
    Returns a list of strings (one per sample in the batch).
    """
    sequences = []
    for i in range(input_ids.shape[0]):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
        # Filter out special tokens
        aa_seq = ""
        for t in tokens:
            if t in STANDARD_AAS:
                aa_seq += t
        sequences.append(aa_seq)
    return sequences


def compute_aac(sequence):
    """Amino Acid Composition: frequency of each of 20 standard AAs."""
    counts = np.zeros(20, dtype=np.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}
    L = len(sequence)
    if L == 0:
        return counts
    for aa in sequence:
        if aa in aa_to_idx:
            counts[aa_to_idx[aa]] += 1
    return counts / L


def compute_dpc(sequence):
    """Dipeptide Composition: frequency of each of 400 possible dipeptides."""
    counts = np.zeros(400, dtype=np.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}
    L = len(sequence)
    if L < 2:
        return counts
    num_dipeptides = 0
    for i in range(L - 1):
        aa1, aa2 = sequence[i], sequence[i + 1]
        if aa1 in aa_to_idx and aa2 in aa_to_idx:
            idx = aa_to_idx[aa1] * 20 + aa_to_idx[aa2]
            counts[idx] += 1
            num_dipeptides += 1
    if num_dipeptides > 0:
        counts /= num_dipeptides
    return counts


def compute_paac(sequence, lamda=15, w=0.05):
    """
    Pseudo Amino Acid Composition (Type I PseAAC).
    Returns a vector of length 20 + lamda*2 = 50 (with lamda=15).
    Uses hydrophobicity (H1) and hydrophilicity (H2) for sequence-order features.
    """
    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}

    # Hydrophobicity values (Kyte-Doolittle, normalized)
    H1_raw = {aa: AAINDEX_TABLE[aa][0] for aa in STANDARD_AAS}
    H2_raw = {aa: PC6_TABLE[aa][0] for aa in STANDARD_AAS}  # PC6 hydrophobicity

    # Normalize H1 and H2
    def _normalize(d):
        vals = np.array(list(d.values()))
        m, s = vals.mean(), vals.std()
        if s < 1e-8:
            s = 1.0
        return {k: (v - m) / s for k, v in d.items()}

    H1 = _normalize(H1_raw)
    H2 = _normalize(H2_raw)

    L = len(sequence)
    # Filter to standard AAs only
    seq_filtered = [aa for aa in sequence if aa in aa_to_idx]
    L_f = len(seq_filtered)

    # AAC part
    aac = np.zeros(20, dtype=np.float32)
    for aa in seq_filtered:
        aac[aa_to_idx[aa]] += 1

    # Sequence-order correlation factors
    actual_lamda = min(lamda, L_f - 1) if L_f > 1 else 0
    theta = np.zeros(lamda * 2, dtype=np.float32)  # lamda values for H1, lamda for H2

    for lag in range(1, actual_lamda + 1):
        sum_h1 = 0.0
        sum_h2 = 0.0
        count = 0
        for i in range(L_f - lag):
            r_i = seq_filtered[i]
            r_j = seq_filtered[i + lag]
            sum_h1 += (H1[r_i] - H1[r_j]) ** 2
            sum_h2 += (H2[r_i] - H2[r_j]) ** 2
            count += 1
        if count > 0:
            theta[lag - 1] = sum_h1 / count           # H1 correlation at this lag
            theta[lamda + lag - 1] = sum_h2 / count    # H2 correlation at this lag

    # Combine: PseAAC = [f1, ..., f20, theta1_H1, ..., theta_lamda_H1, theta1_H2, ..., theta_lamda_H2]
    denom = aac.sum() + w * theta.sum()
    if denom < 1e-8:
        denom = 1.0

    paac_vec = np.zeros(20 + lamda * 2, dtype=np.float32)
    paac_vec[:20] = aac / denom
    paac_vec[20:] = w * theta / denom

    return paac_vec


def compute_global_features_batch(sequences):
    """
    Compute AAC + DPC + PseAAC for a batch of amino acid sequences.
    Returns a tensor of shape [B, GLOBAL_FEATURE_DIM].
    """
    batch_feats = []
    for seq in sequences:
        aac = compute_aac(seq)
        dpc = compute_dpc(seq)
        paac = compute_paac(seq, lamda=15, w=0.05)
        feat = np.concatenate([aac, dpc, paac])
        batch_feats.append(feat)
    return torch.from_numpy(np.stack(batch_feats, axis=0))  # [B, 470]



class PhysicochemicalBranch(nn.Module):
    """
    Physicochemical feature extractor following the Pep2Net paper:
    - Residue-level features: AAindex (12-dim) + PC6 (6-dim) + BLOSUM62 (20-dim) = 38-dim per residue
    - Global features: AAC (20-dim) + DPC (400-dim) + PseAAC (50-dim) = 470-dim per sequence
    - 1D CNN for local physicochemical motifs
    - BiLSTM for global physicochemical sequence context
    """
    def __init__(self, tokenizer, hidden_dim=256, out_dim=768, dropout=0.2):
        super().__init__()
        self.tokenizer = tokenizer

        # Fixed residue-level feature lookup table (NOT learned)
        residue_table = _build_residue_feature_table(tokenizer)
        self.register_buffer('residue_feature_table', residue_table)  # [vocab_size, 38]

        # Projection for global features to per-position
        self.global_proj = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- MODIFIED: three independent projections, one per feature group ---
        self.proj_pc6     = nn.Linear(PC6_DIM,      hidden_dim)
        self.proj_aaindex = nn.Linear(AAINDEX_DIM,  hidden_dim)
        self.proj_blosum  = nn.Linear(BLOSUM62_DIM, hidden_dim)
        # Fuse concat(hidden_dim * 3) back to hidden_dim so downstream dims are unchanged
        self.residue_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # --- END MODIFIED ---

        # 1D CNN for local physicochemical motifs (operates on combined features)
        combined_input_dim = hidden_dim * 2  # residue_proj + global_proj  (unchanged)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=combined_input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # BiLSTM for global physicochemical sequence context
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, L] - ESM tokenized input
        attention_mask: [B, L] - 1 for valid tokens, 0 for padding
        Returns: [B, L, out_dim]
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 1. Residue-level features from fixed lookup table
        clamped_ids = input_ids.clamp(0, self.residue_feature_table.shape[0] - 1)
        raw = self.residue_feature_table[clamped_ids]  # [B, L, 38]

        # --- MODIFIED: split by group, project independently, concat, then fuse ---
        pc6_feat     = raw[..., :PC6_DIM]                             # [B, L, 6]
        aaindex_feat = raw[..., PC6_DIM:PC6_DIM + AAINDEX_DIM]        # [B, L, 12]
        blosum_feat  = raw[..., PC6_DIM + AAINDEX_DIM:]               # [B, L, 20]

        residue_feats = torch.cat([
            self.proj_pc6(pc6_feat),         # [B, L, hidden_dim]
            self.proj_aaindex(aaindex_feat),  # [B, L, hidden_dim]
            self.proj_blosum(blosum_feat),    # [B, L, hidden_dim]
        ], dim=-1)                            # [B, L, hidden_dim * 3]
        residue_feats = self.residue_fusion(residue_feats)  # [B, L, hidden_dim]
        # --- END MODIFIED ---

        # 2. Global composition features (AAC + DPC + PseAAC)
        aa_sequences = _token_ids_to_aa_sequences(input_ids, self.tokenizer)
        global_feats = compute_global_features_batch(aa_sequences).to(device)  # [B, 470]
        global_feats = self.global_proj(global_feats)            # [B, hidden_dim]
        # Broadcast global features to every position
        global_feats = global_feats.unsqueeze(1).expand(-1, L, -1)  # [B, L, hidden_dim]

        # 3. Combine residue-level and global features
        combined = torch.cat([residue_feats, global_feats], dim=-1)  # [B, L, hidden_dim*2]

        # 4. 1D CNN for local motifs
        x = combined.transpose(1, 2)      # [B, hidden_dim*2, L]
        x = self.cnn(x)                   # [B, hidden_dim, L]
        x = x.transpose(1, 2)             # [B, L, hidden_dim]

        # 5. BiLSTM for sequential context
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.bilstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=L
            )
        else:
            x, _ = self.bilstm(x)

        x = self.dropout(x)

        # 6. Project to output dimension
        x = self.proj(x)                  # [B, L, out_dim]
        return x


class DualFusionBlock(nn.Module):
    """
    Implements Equations 3-8 from the Pep2Net paper:
    Cross-Attention Alignment + Low-Rank Bilinear Fusion.
    """
    def __init__(self, d_model, dh=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.dh = dh
        self.n_heads = n_heads

        # Eq 3: Projection to common latent space
        self.Ws = nn.Linear(d_model, dh)
        self.Wp = nn.Linear(d_model, dh)

        # Eq 4 & 5: Cross Attention Layers
        self.cross_attn_S = nn.MultiheadAttention(embed_dim=dh, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_P = nn.MultiheadAttention(embed_dim=dh, num_heads=n_heads, dropout=dropout, batch_first=True)

        # Eq 6 & 7: Low Rank Bilinear Pooling FFN
        self.bilinear_ffn = nn.Sequential(
            nn.Linear(dh, dh * 2),
            nn.LayerNorm(dh * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dh * 2, dh),
            nn.LayerNorm(dh),
        )

        self.layer_norm_s = nn.LayerNorm(dh)
        self.layer_norm_p = nn.LayerNorm(dh)

    def forward(self, S, P, attention_mask):
        """
        S: Semantic embeddings [B, L, d_model]
        P: Physicochemical embeddings [B, L, d_model]
        attention_mask: [B, L] (1 for valid, 0 for pad)
        """
        key_padding_mask = (attention_mask == 0)

        # Eq 3: Project
        S_prime = self.Ws(S)  # [B, L, dh]
        P_prime = self.Wp(P)  # [B, L, dh]

        # Eq 4 & 5: Cross-Attention Alignment
        S_attn, _ = self.cross_attn_S(query=S_prime, key=P_prime, value=P_prime, key_padding_mask=key_padding_mask)
        S_attn = self.layer_norm_s(S_prime + S_attn)

        P_attn, _ = self.cross_attn_P(query=P_prime, key=S_prime, value=S_prime, key_padding_mask=key_padding_mask)
        P_attn = self.layer_norm_p(P_prime + P_attn)

        # Eq 6 & 7: Low-Rank Bilinear Interaction
        Z = S_prime * P_prime  # Hadamard product [B, L, dh]

        # Masked Pooling for Z
        mask_expanded = attention_mask.unsqueeze(-1).float()
        Z_sum = torch.sum(Z * mask_expanded, dim=1)
        Z_lengths = attention_mask.sum(1).unsqueeze(-1).clamp(min=1e-9)
        Z_pooled = Z_sum / Z_lengths  # [B, dh]

        v_bilinear = self.bilinear_ffn(Z_pooled)  # [B, dh]

        return S_attn, P_attn, v_bilinear


class EsmForMultiLabelSequenceClassification(nn.Module):
    def __init__(self, esm_pretrained_model_dir=None, num_labels=21,
                 fusion_dim=256, contrastive_dim=256, dropout=0.2, tokenizer=None):
        super(EsmForMultiLabelSequenceClassification, self).__init__()

        self.num_labels = num_labels

        # 1. Semantic Branch (ESM)
        self.esm = EsmModel.from_pretrained(esm_pretrained_model_dir, output_attentions=True)
        d_model = self.esm.config.hidden_size

        # 2. Physicochemical Branch (Fixed features + 1D CNN + BiLSTM)
        if tokenizer is None:
            from transformers import EsmTokenizer
            tokenizer = EsmTokenizer.from_pretrained(esm_pretrained_model_dir)
        self.physico_branch = PhysicochemicalBranch(
            tokenizer=tokenizer, hidden_dim=256, out_dim=d_model, dropout=dropout
        )

        # 3. Dual Fusion Block (Cross-Attention + Bilinear)
        self.fusion_block = DualFusionBlock(d_model=d_model, dh=fusion_dim, n_heads=8, dropout=dropout)

        # 4. Multi-Sample Dropout Head (Eq 8)
        concat_dim = (fusion_dim * 3) + (d_model * 2)
        self.msd_head = MultiSampleDropoutHead(num_labels=self.num_labels, hidden_dim=concat_dim, p=0.5, n=6)

        # 5. Contrastive Projection Head
        self.contrastive_dim = contrastive_dim
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, contrastive_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(contrastive_dim, contrastive_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(self.num_labels, contrastive_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, cb_weights=None, **kwargs):

        # 1. Semantic Features
        esm_out = self.esm(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        S = esm_out.last_hidden_state  # [B, L, d_model]
        attention_matrix = esm_out.attentions[-1]

        # 2. Physicochemical Features (using fixed descriptors)
        P = self.physico_branch(input_ids, attention_mask)  # [B, L, d_model]

        # 3. Dual Fusion (Eq 3 - 7)
        S_attn, P_attn, v_bilinear = self.fusion_block(S, P, attention_mask)

        # 4. Global Pooling (Masked Mean)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_mask = attention_mask.sum(1).unsqueeze(-1).clamp(min=1e-9)

        S_global = torch.sum(S * mask_expanded, 1) / sum_mask           # [B, d_model]
        P_global = torch.sum(P * mask_expanded, 1) / sum_mask           # [B, d_model]
        S_attn_pool = torch.sum(S_attn * mask_expanded, 1) / sum_mask   # [B, dh]
        P_attn_pool = torch.sum(P_attn * mask_expanded, 1) / sum_mask   # [B, dh]

        # Eq 8: Final concatenation
        F_final = torch.cat([S_attn_pool, P_attn_pool, v_bilinear, S_global, P_global], dim=-1)

        # 5. Classification
        prediction_res = self.msd_head(F_final)  # [B, num_labels]

        # 6. Contrastive embedding
        z = self.proj(F_final)                   # [B, contrastive_dim]
        z = F.normalize(z, p=2, dim=-1)
        pred_emb = z

        if labels is not None:
            loss_fn = ClassBalancedFocalDiceLoss(cb_weights=cb_weights)
            loss_decoder = loss_fn(prediction_res, labels)
            return loss_decoder, prediction_res, pred_emb, attention_matrix
        else:
            return prediction_res, pred_emb, attention_matrix


# =========================================================================================
#  Helper Classes
# =========================================================================================

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class MultiSampleDropoutHead(nn.Module):
    def __init__(self, num_labels, hidden_dim, p=0.5, n=6):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(n)])
        self.classifier = GroupWiseLinear(num_labels, hidden_dim)
        self.n = n

    def forward(self, x):
        logits = 0
        for d in self.dropouts:
            logits = logits + self.classifier(d(x))
        return logits / self.n  # [B, num_labels]


class ClassBalancedFocalDiceLoss(nn.Module):
    """
    Combines Focal Dice Loss with Class-Balanced Weights.
    """
    def __init__(self, p_pos=2, p_neg=3, clip_pos=0.7, clip_neg=0.5, pos_weight=0.2, reduction='mean', cb_weights=None):
        super(ClassBalancedFocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight
        self.cb_weights = cb_weights

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0]
        predict = nn.Sigmoid()(input).contiguous().view(input.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Positives
        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)
        loss_pos = 1 - (2 * num_pos) / den_pos

        # Negatives
        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)
        loss_neg = 1 - (2 * num_neg) / den_neg

   
        loss_pos = (1 - (2 * num_pos) / den_pos).clamp(min=0)
        loss_neg = (1 - (2 * num_neg) / den_neg).clamp(min=0)

        # Combine
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)

        # Apply Class-Balanced weights if provided
        if self.cb_weights is not None:
            sample_weights = torch.matmul(target.float(), self.cb_weights.to(target.device))
            sample_weights = sample_weights.clamp(min=1.0)
            loss = loss * sample_weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
