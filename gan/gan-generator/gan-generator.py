import numpy as np

def generator(z, W, b):
    """
    Returns: np.ndarray of shape (batch, output_dim) with tanh-activated values rounded to 4 decimals
    """
    z, W, b = map(np.asarray, [z, W, b])
    G = np.tanh(z @ W + b)
    return G