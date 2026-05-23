import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    assert y_left is not None and y_right is not None
    y_left, y_right = np.asarray(y_left), np.asarray(y_right)
    N_L, N_R = y_left.shape[0], y_right.shape[0]
    N = N_L + N_R
    if N == 0:
        return 0.
    # classes = np.unique(np.concatenate([y_left, y_right], axis=0))
    def gini_node(y):
        if y.shape[0] == 0:
            return 0.
        _, counts = np.unique(y, axis=0, return_counts=True)
        prob = counts / y.shape[0]
        return 1. - np.sum(prob ** 2)

    # assert False, y_left_perclasses

    gini_left = gini_node(y_left)
    gini_right = gini_node(y_right)
    return (N_L * gini_left + N_R * gini_right) / N
    # return None