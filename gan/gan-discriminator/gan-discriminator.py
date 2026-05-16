import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x))

def discriminator(x, W):
    """
    Returns: np.ndarray of shape (batch, 1) with probabilities rounded to 4 decimals
    """
    x, W = map(np.asarray, [x, W])
    return sigmoid(x @ W)