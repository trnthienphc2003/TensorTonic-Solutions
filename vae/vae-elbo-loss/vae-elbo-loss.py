import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    # Your implementation here
    recon = np.mean(np.sum((x_recon - x) ** 2, axis=-1), axis=0)

    var = np.exp(log_var)
    kl = np.mean(np.sum(-0.5 * (1 + log_var - mu ** 2 - var), axis=-1), axis=0)
    return {
        'total': recon + kl,
        'recon': recon,
        'kl': kl,
    }
