import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    pass
    n = len(v)
    mat = np.zeros((n, n))
    for i in range(len(v)): mat[i][i] = v[i];
    return mat;