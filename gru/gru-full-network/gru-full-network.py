import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def gru_cell(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Complete GRU cell forward pass.
        """
        # YOUR CODE HERE
        state_t = np.concatenate([h_prev, x_t], axis=-1)
        r_t = sigmoid(state_t @ self.W_r.T + self.b_r)
        z_t = sigmoid(state_t @ self.W_z.T + self.b_z)
    
        mem_t = np.concatenate([r_t * h_prev, x_t], axis=-1)
        h_tilde_t = np.tanh(mem_t @ self.W_h.T + self.b_h)
        h_t = z_t * h_prev + (1 - z_t) * h_tilde_t
        return h_t

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last).
        """
        # YOUR CODE HERE
        B, T, input_dim = X.shape
        y = []
        h_cur = np.zeros((B, self.hidden_dim))

        for t in range(T):
            h_cur = self.gru_cell(X[:, t, :], h_cur)
            y.append(h_cur @ self.W_y.T + self.b_y)

        return np.stack(y, axis=1), h_cur