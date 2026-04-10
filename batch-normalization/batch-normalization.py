import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    pass
    x = np.array(x)
    gamma, beta = np.array(gamma), np.array(beta)
    if x.ndim == 2:
        mean = np.average(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        new_x = (x - mean) / np.sqrt(var + eps)
        y = gamma * new_x + beta
        return y

    N, C, H, W = x.shape
    mean = np.average(x, axis=(0, 2, 3), keepdims=True)
    var = np.var(x, axis=(0, 2, 3), keepdims=True)
    new_x = (x - mean) / np.sqrt(var + eps)
    gamma = np.reshape(gamma, (1, -1, 1, 1))
    beta = np.reshape(beta, (1, -1, 1, 1))
    y = gamma * new_x + beta
    return y