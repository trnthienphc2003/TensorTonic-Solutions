import numpy as np

def compute_ddpm_loss(x_0, betas, t_values, epsilon, epsilon_pred):
    """
    Returns: float scalar MSE loss between true noise and predicted noise
    """
    # YOUR CODE HERE
    epsilon, epsilon_pred = np.asarray(epsilon, dtype=np.float64), np.asarray(epsilon_pred, dtype=np.float64)

    return np.mean(np.abs(epsilon - epsilon_pred) ** 2)