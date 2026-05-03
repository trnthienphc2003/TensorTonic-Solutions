import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    y = np.asarray(y)
    assert (y.ndim == 1)
    if num_classes is None:
        num_classes = np.max(y) + 1

    n = y.shape[0]
    
    one_hot_matrix = np.full((n, num_classes), np.arange(num_classes))
    y = y.reshape((n, 1))
    # assert False, y
    # assert False, np.where(one_hot_matrix == y, 1, 0)
    return np.where(one_hot_matrix == y, 1, 0)