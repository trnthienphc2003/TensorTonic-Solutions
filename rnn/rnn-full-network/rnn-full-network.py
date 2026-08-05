import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        B, T, input_dim = X.shape
        if h_0 is None:
            h_0 = np.zeros((B, self.hidden_dim, self.hidden_dim))

        h_cur = None
        y_full = []
        for t in range(T):
            h_cur = np.tanh(X[:, t, :] @ self.W_xh.T + h_0 @ self.W_hh.T + self.b_h)
            y_cur = h_cur @ self.W_hy.T + self.b_y

            h_0 = h_cur[:]
            y_full.append(y_cur)

        return np.stack(y_full, axis=1), h_0