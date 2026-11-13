import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# BLOCK 11: UNCERTAINTY-AWARE LOSS
# ------------------------------------------------------------
# UPDATED:
# - Added Label Smoothing
# - Improves calibration
# - Reduces overconfidence
# - Helps hard-sample generalization
# ============================================================

def uncertainty_loss(outputs, targets, sigma_r, sigma_m):

    # ========================================================
    # LABEL-SMOOTHED CROSS ENTROPY
    # --------------------------------------------------------
    # label_smoothing = 0.1 means:
    #
    # correct class → 0.9
    # remaining probability distributed softly
    # ========================================================

    ce = F.cross_entropy(
        outputs,
        targets,
        label_smoothing=0.1
    )

    # ========================================================
    # GLOBAL UNCERTAINTY
    # ========================================================

    sigma = (sigma_r.mean() + sigma_m.mean()) / 2

    # ========================================================
    # UNCERTAINTY-AWARE LOSS
    # ========================================================

    loss = ce * torch.exp(-sigma) + sigma

    return loss

#Metric Computation 
def compute_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1
