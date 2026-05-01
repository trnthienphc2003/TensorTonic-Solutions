import numpy as np

def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    
    Args:
        x: np.ndarray (1D or 2D)
        axis: axis to apply softmax over (default: last axis)
    
    Returns:
        np.ndarray with same shape as x
    """
    x = np.asarray(x, dtype=np.float64)

    # subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max

    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)

    return exp_x / sum_exp

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    pass
    Z1, Z2 = np.asarray(Z1), np.asarray(Z2)
    assert Z1.ndim == 2
    assert Z2.ndim == 2

    N, D = Z1.shape
    S = (Z1 @ Z2.T) / temperature

    logits_max = np.max(S, axis=-1, keepdims=True)
    logits_shifted = S - logits_max
    log_probs = logits_shifted - np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))

    # positives are diagonal
    return -np.mean(np.diag(log_probs))