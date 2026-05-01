import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    diff = np.abs(y_true - y_pred)
    # assert False, diff
    return np.average(np.where(diff > delta, delta * (diff - .5 * delta), .5 * diff * diff))