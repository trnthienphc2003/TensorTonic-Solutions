import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    pass
    try:
        return np.linalg.inv(A)
    except Exception as e:
        # raise e
        pass
        # return None
    return None