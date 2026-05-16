import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE
    betas = np.asarray(betas)
    return np.cumprod(1. - betas)

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    # YOUR CODE HERE
    x_0, betas, epsilon = map(np.asarray, [x_0, betas, epsilon])
    alpha_bar = get_alpha_bar(betas)

    # assert False, alpha_bar
    return np.sqrt(alpha_bar[t - 1]) * x_0 + epsilon * np.sqrt(1. - alpha_bar[t - 1])