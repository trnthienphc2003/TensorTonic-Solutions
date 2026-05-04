import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Write code here
    real_scores, fake_scores = np.asarray(real_scores, dtype=np.float64), np.asarray(fake_scores, dtype=np.float64)

    # return np.mean(1. / (1. + np.exp(-fake_scores))) - np.mean(1. / (1. + np.exp(-real_scores)))
    return np.mean(fake_scores) - np.mean(real_scores)