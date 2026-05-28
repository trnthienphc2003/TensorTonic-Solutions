import numpy as np
from sklearn.model_selection import train_test_split

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split X and y into train/test while preserving class proportions.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)
    num_classes = len(classes)
    train_indices, test_indices = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if rng is None:
            np.random.shuffle(idx)
        else:
            rng.shuffle(idx)

        N = len(idx)
        n_test = round(N * test_size)
        n_train = N - n_test

        if n_train == 0:
            n_train = 1
            n_test = N - 1

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    train_indices.sort()
    test_indices.sort()
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]