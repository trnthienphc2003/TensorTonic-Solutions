import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    B, T, input_dim = X.shape
    hidden_dim = W_xh.shape[0]
    h_prev = h_0[:]
    h_full = np.zeros((B, T, hidden_dim))
    for t in range(T):
        h_new = np.tanh(X[:, t, :] @ W_xh.T + h_prev @ W_hh.T + b_h)
        h_full[:, t, :] = h_new[:]
        h_prev = h_new[:]

    return h_full, h_prev