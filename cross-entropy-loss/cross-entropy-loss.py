import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    # y_true: (n_samples, )
    # y_pred: (n_samples, n_classes)
    n_samples = len(y_true)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred_care = np.log(y_pred[np.arange(n_samples), y_true])
    # assert False, 
    
    return -np.average(y_pred_care)