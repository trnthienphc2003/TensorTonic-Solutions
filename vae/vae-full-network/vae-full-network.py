import numpy as np

class VAE:
    def __init__(self, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray):
        """
        Initialize VAE with concrete weight matrices.
        """
        # Store weights here
        self.W_mu = W_mu
        self.b_mu = b_mu
        self.W_logvar = W_logvar
        self.b_logvar = b_logvar
        self.W_dec = W_dec
        self.b_dec = b_dec
    
    def forward(self, x: np.ndarray, epsilon: np.ndarray) -> dict:
        """
        Full forward pass: encode -> reparameterize -> decode.
        Returns dict with "recon", "mu", "log_var".
        """
        # Your implementation here
        # encode
        x_enc_mu = x @ self.W_mu + self.b_mu
        x_enc_logvar = x @ self.W_logvar + self.b_logvar

        # Reparameterize
        sigma = np.exp(.5 * x_enc_logvar)
        z = x_enc_mu + sigma * epsilon

        # Decode
        x_recon = z @ self.W_dec + self.b_dec
        mu = x_enc_mu @ self.W_dec + self.b_dec
        log_var = x_enc_logvar @ self.W_dec + self.b_dec
        return {
            'recon': x_recon,
            'mu': x_enc_mu,
            'log_var': x_enc_logvar
        }
    
    def generate(self, z: np.ndarray) -> np.ndarray:
        """
        Generate samples from given latent vectors.
        """
        # Your implementation here
        out = z @ self.W_dec + self.b_dec
        return out
