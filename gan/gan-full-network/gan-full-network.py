import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x))

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        # z, G_W = map(np.asarray, [z, G_W])
        z = np.asarray(z)
        return np.tanh(z @ self.G_W)
    
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        x = np.asarray(x)
        return sigmoid(x @ self.D_W)
    def discriminator_loss(self, real_probs, fake_probs):
        """Compute discriminator loss using binary cross-entropy.
        Returns: Loss value rounded to 4 decimals."""
        eps = 1e-8
        real_probs, fake_probs = map(np.asarray, [real_probs, fake_probs])
        return -np.average(np.log(np.clip(real_probs, a_min=eps, a_max=1. - eps))) - np.average(np.log(np.clip(1. - fake_probs, a_min=eps, a_max=1. - eps)))
    
    def generator_loss(self, fake_probs):
        """Compute non-saturating generator loss.
        Returns: Loss value rounded to 4 decimals."""
        # pass
        eps = 1e-8
        fake_probs = np.asarray(fake_probs)
        return -np.average(np.log(np.clip(fake_probs, a_min=eps, a_max=1. - eps)))
    
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        # Your implementation here
        fake_data = self.generate(z)
        # real_data = self.discriminate(real_data)

        fake_probs = self.discriminate(fake_data)
        real_probs = self.discriminate(real_data)
        return {
            'd_loss': self.discriminator_loss(real_probs, fake_probs),
            'g_loss': self.generator_loss(fake_probs)
        }
        