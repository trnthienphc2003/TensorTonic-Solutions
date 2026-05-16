import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x))

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    eps = 1e-8
    real_probs, fake_probs = map(np.asarray, [real_probs, fake_probs])
    return -np.average(np.log(np.clip(real_probs, a_min=eps, a_max=1. - eps))) - np.average(np.log(np.clip(1. - fake_probs, a_min=eps, a_max=1. - eps)))

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    # pass
    eps = 1e-8
    fake_probs = np.asarray(fake_probs)
    return -np.average(np.log(np.clip(fake_probs, a_min=eps, a_max=1. - eps)))

def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    # Your implementation here
    real_data, fake_data, D_W = map(np.asarray, [real_data, fake_data, D_W])

    real_probs = sigmoid(real_data @ D_W)
    fake_probs = sigmoid(fake_data @ D_W)
    return {
        'd_loss': discriminator_loss(real_probs, fake_probs),
        'g_loss': generator_loss(fake_probs)
    }