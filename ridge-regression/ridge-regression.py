import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """

    X, y = np.asarray(X), np.asarray(y)
    N, D = X.shape
    reg_mat = X.T @ X + lam * np.eye(D)
    w = np.linalg.inv(reg_mat) @ (X.T @ y)
    return w
    