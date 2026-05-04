import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X, y = np.asarray(X), np.asarray(y)
    return np.linalg.inv(X.T @ X) @ X.T @ y