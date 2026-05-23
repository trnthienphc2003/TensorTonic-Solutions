import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left, y_right = np.asarray(y_left), np.asarray(y_right)
    N_L, N_R = y_left.shape[0], y_right.shape[0]
    N = N_L + N_R
    if N == 0:
        return 0.
    classes = np.unique(np.concatenate([y_left, y_right], axis=0))
    def gini_node(y, classes):
        if np.any(y):
            y_onehot = y[..., np.newaxis] == classes[np.newaxis, ...]
            y_perclass = np.average(y_onehot, axis=0)
            return 1. - np.sum(y_perclass ** 2)
        return 0.

    # assert False, y_left_perclasses

    gini_left = gini_node(y_left, classes)
    gini_right = gini_node(y_right, classes)
    return (N_L * gini_left + N_R * gini_right) / N
    # return None