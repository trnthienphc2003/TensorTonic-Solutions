import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def lstm_cell(self, x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray) -> tuple:
        """Complete LSTM cell forward pass."""
        state_cur = np.concatenate([h_prev, x_t], axis=-1) # (N, H + D)
        f_t = sigmoid(state_cur @ self.W_f.T + self.b_f) # (N, H)
        i_t = sigmoid(state_cur @ self.W_i.T + self.b_i)
        C_tilde = np.tanh(state_cur @ self.W_c.T + self.b_c)
        o_t = sigmoid(state_cur @ self.W_o.T + self.b_o)
    
        C_t = f_t * C_prev + i_t * C_tilde
        h_t = o_t * np.tanh(C_t)
    
        return h_t, C_t

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last, C_last).
        """
        # YOUR CODE HERE
        B, T, _ = X.shape
        h_cur, C_cur = np.zeros((B, self.hidden_dim)), np.zeros((B, self.hidden_dim))
        y = []
        for t in range(T):
            h_cur, C_cur = self.lstm_cell(X[:, t, :], h_cur, C_cur)
            out_y = h_cur @ self.W_y.T + self.b_y
            y.append(out_y)
            # assert False, out_y.shape 
        return np.stack(y, axis=1), h_cur, C_cur