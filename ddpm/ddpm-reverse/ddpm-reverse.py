import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    # YOUR CODE HERE
    x_t, betas = np.asarray(x_t, dtype=np.float64), np.asarray(betas, dtype=np.float64)
    epsilon_pred = np.asarray(epsilon_pred, dtype=np.float64)

    beta_t = betas[t - 1]

    alpha_t = 1. - beta_t
    alpha_bar_t = np.cumprod(1. - betas, axis=-1)[t - 1]
    sigma = np.sqrt(beta_t)

    x = 1. / np.sqrt(alpha_t) * (x_t - epsilon_pred * (1. - alpha_t) / (np.sqrt(1. - alpha_bar_t)))
    if t != 1:
        assert z is not None
        x += sigma * np.asarray(z, dtype=np.float64)
    return x