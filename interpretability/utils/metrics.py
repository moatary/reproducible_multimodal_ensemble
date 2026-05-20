
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def compute_f1mod(y_true, y_pred, m=None):
    """Modified F1 (Equation 3 of the paper).

    F1-mod = TP / (TP + 0.5*FP/m + 0.5*FN)

    where m = neg/pos ratio (>= 4 required for metric to differ from F1)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = ((y_pred==1)&(y_true==1)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()
    if m is None:
        m = max((y_true==0).sum() / max((y_true==1).sum(), 1e-7), 4.0)
    denom = tp + 0.5*fp/m + 0.5*fn
    return float(tp / denom) if denom > 0 else 0.0
