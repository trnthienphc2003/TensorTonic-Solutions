import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a, b, y = np.asarray(a), np.asarray(b), np.asarray(y)
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    diff = np.linalg.norm(b - a, axis=-1)

    # assert False, diff
    # val = y * diff * diff

    l = y * diff * diff + (1 - y) * np.pow(np.maximum(0, margin - diff), 2)
    if reduction == 'mean':
        return np.mean(l)
    return np.sum(l)
    # return 0.