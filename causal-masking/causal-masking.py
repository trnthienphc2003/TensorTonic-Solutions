import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    scores = np.asarray(scores)
    T = scores.shape[-1]
    return (np.tril(scores, 0) + np.triu(np.full((1, T, T), mask_value), 1)).reshape(scores.shape)