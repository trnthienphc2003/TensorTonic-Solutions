import numpy as np

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