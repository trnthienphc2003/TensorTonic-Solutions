import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    matrix = np.array(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        # raise ValueError("The matrix must be 2-dimension")
        return None

    if axis is not None and (axis > 1 or axis < 0):
        return None

    if norm_type == 'l1':
        ord_val = 1
    elif norm_type == 'l2':
        ord_val = None if axis is None else 2
    elif norm_type == 'max':
        ord_val = np.inf
    else:
        # raise ValueError("Unsupported norm_type")
        return None

    norm = np.linalg.norm(matrix, ord=ord_val, axis=axis, keepdims=True)

    if np.any(norm == 0):
        norm = np.where(norm == 0, 1, norm)

    return matrix / norm