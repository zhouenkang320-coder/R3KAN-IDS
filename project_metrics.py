import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def strict_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.size == 0:
        return 0.0
    unique = np.unique(labels)
    if len(unique) == 1:
        return float(unique[0] == 1)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))
