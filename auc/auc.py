import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr, tpr = np.array(fpr), np.array(tpr)
    t = np.delete(np.roll(tpr, -1) + tpr, -1, 0)
    f = np.delete(np.roll(fpr, -1) - fpr, -1, 0)
    return .5 * np.dot(t, f)