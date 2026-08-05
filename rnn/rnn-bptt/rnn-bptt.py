import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # YOUR CODE HERE
    B, hidden_dim = dh_next.shape
    dh_prev = np.zeros((B, hidden_dim))
    dW_hh = np.zeros((hidden_dim, hidden_dim))

    d_tanh = (1 - h_t ** 2) * dh_next
    dh_prev = d_tanh @ W_hh
    dW_hh = d_tanh.T @ h_prev
    return dh_prev, dW_hh