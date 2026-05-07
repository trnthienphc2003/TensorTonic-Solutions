import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2 or X.shape[0] < 2:
        return None

    X_centered = X - X.mean(axis=0, keepdims=True)
    std = X_centered.std(axis=0, ddof=1, keepdims=True)


    Z = X_centered / std

    return (Z.T @ Z) / (X.shape[0] - 1)
    