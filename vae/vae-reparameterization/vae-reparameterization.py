import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray z of shape (batch, latent_dim) sampled via reparameterization
    """
    # Your implementation here
    pass
    sigma = np.exp(0.5 * log_var)
    z = mu + sigma * epsilon
    return z