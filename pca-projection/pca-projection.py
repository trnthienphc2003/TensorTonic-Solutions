import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.asarray(X)

    Xc = (X - X.mean(axis=0))
    # assert False, Xc
    N, D = Xc.shape
    C = 1./(N - 1) * (Xc.T @ Xc)
    # assert False, C

    eig_values, eig_vectors = np.linalg.eig(C)
    # assert False, eig_values.shape
    # assert False, eig_vectors.shape
    idx = np.argsort(eig_values)[::-1]
    # assert False, f'eigenvalues: {eig_values} idx: {idx}'
    W = eig_vectors[:, idx[:k]]

    X_proj = Xc @ W
    return X_proj