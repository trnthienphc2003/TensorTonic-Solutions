import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Write code here
    X = np.asarray(X)
    n, d = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_c)
    
    
    # (N, N), (N, D), (D, D)
    return X_c @ (Vt.T[..., :k])

    # S[k:] = 0
    # assert False, Vt.shape
    # W = X @ Vt.T[..., :k]
    # return X_c @ W