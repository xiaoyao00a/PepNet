import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    auc,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    precision_recall_curve
)


def scores(y_test, y_pred, th=0.5):

    y_test = np.asarray(y_test).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    y_predlabel = np.array([(0 if item < th else 1) for item in y_pred], dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).ravel()

    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_test, y_predlabel) if len(np.unique(y_test)) > 1 else 0.0
    recall = recall_score(y_test, y_predlabel, zero_division=0)
    precision = precision_score(y_test, y_predlabel, zero_division=0)
    f1 = f1_score(y_test, y_predlabel, zero_division=0)
    acc = accuracy_score(y_test, y_predlabel)
    auc_score = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.0

    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    aupr = auc(recall_aupr, precision_aupr)

    return [recall, spe, precision, f1, mcc, acc, auc_score, aupr, tp, fn, tn, fp]


def _safe_div(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)


def Aiming(y_hat, y):
    """
    sample-based precision
    """
    y_hat = np.asarray(y_hat).astype(int)
    y = np.asarray(y).astype(int)

    tp = ((y_hat == 1) & (y == 1)).sum(axis=1).astype(np.float64)
    fp = ((y_hat == 1) & (y == 0)).sum(axis=1).astype(np.float64)
    return _safe_div(tp, tp + fp).mean()


def Coverage(y_hat, y):
    """
    sample-based recall
    """
    y_hat = np.asarray(y_hat).astype(int)
    y = np.asarray(y).astype(int)

    tp = ((y_hat == 1) & (y == 1)).sum(axis=1).astype(np.float64)
    fn = ((y_hat == 0) & (y == 1)).sum(axis=1).astype(np.float64)
    return _safe_div(tp, tp + fn).mean()


def Accuracy(y_hat, y):
    """
    sample-based accuracy / Jaccard
    """
    y_hat = np.asarray(y_hat).astype(int)
    y = np.asarray(y).astype(int)

    tp = ((y_hat == 1) & (y == 1)).sum(axis=1).astype(np.float64)
    fp = ((y_hat == 1) & (y == 0)).sum(axis=1).astype(np.float64)
    fn = ((y_hat == 0) & (y == 1)).sum(axis=1).astype(np.float64)
    return _safe_div(tp, tp + fp + fn).mean()


def AbsoluteTrue(y_hat, y):
    """
    exact match ratio
    """
    y_hat = np.asarray(y_hat).astype(int)
    y = np.asarray(y).astype(int)
    return np.mean(np.all(y_hat == y, axis=1).astype(np.float64))


def AbsoluteFalse(y_hat, y):
    """
    hamming loss style
    """
    y_hat = np.asarray(y_hat).astype(int)
    y = np.asarray(y).astype(int)

    fp = ((y_hat == 1) & (y == 0)).sum(axis=1).astype(np.float64)
    fn = ((y_hat == 0) & (y == 1)).sum(axis=1).astype(np.float64)
    m = y.shape[1]
    return np.mean((fp + fn) / m)


def evaluate(y_hat, y):
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    return aiming, coverage, accuracy, absolute_true, absolute_false


def multilabel_paper_metrics(y_prob, y_true, th=0.5):

    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    if y_prob.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_prob {y_prob.shape} vs y_true {y_true.shape}")

    if y_true.size == 0:
        return {
            'pred_bin': np.zeros_like(y_true, dtype=np.int64),
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

    y_hat = (y_prob >= th).astype(np.int64)

    tp = ((y_hat == 1) & (y_true == 1)).sum(axis=1).astype(np.float64)
    fp = ((y_hat == 1) & (y_true == 0)).sum(axis=1).astype(np.float64)
    fn = ((y_hat == 0) & (y_true == 1)).sum(axis=1).astype(np.float64)
    tn = ((y_hat == 0) & (y_true == 0)).sum(axis=1).astype(np.float64)

    precision = _safe_div(tp, tp + fp).mean()
    recall = _safe_div(tp, tp + fn).mean()
    specificity = _safe_div(tn, tn + fp).mean()
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn).mean()
    acc = _safe_div(tp, tp + fp + fn).mean()

    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, mcc_den).mean()

    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(y_hat, y_true)

    auc_list = []
    for i in range(y_true.shape[1]):
        yi = y_true[:, i]
        pi = y_prob[:, i]
        if len(np.unique(yi)) < 2:
            continue
        try:
            auc_list.append(roc_auc_score(yi, pi))
        except Exception:
            pass
    auc_macro = float(np.mean(auc_list)) if len(auc_list) > 0 else 0.0

    flat_true = y_true.reshape(-1)
    flat_prob = y_prob.reshape(-1)
    if len(np.unique(flat_true)) >= 2:
        try:
            auc_micro = float(roc_auc_score(flat_true, flat_prob))
        except Exception:
            auc_micro = 0.0
    else:
        auc_micro = 0.0

    return {
        'pred_bin': y_hat,
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
        'auc_macro': float(auc_macro),
        'auc_micro': float(auc_micro),
    }