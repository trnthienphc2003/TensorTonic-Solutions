import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_pred = np.clip(y_pred, eps, 1. - eps)
    p = -y_true * (np.log(y_pred)) - (1 - y_true) * (np.log(1 - y_pred))
    # assert False, p
    return p.tolist()