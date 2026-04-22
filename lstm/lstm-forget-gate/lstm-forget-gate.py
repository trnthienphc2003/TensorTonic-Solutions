import numpy as np

def sigmoid(x):
    # Using np.clip to prevent overflow in the exp function
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """
    Compute forget gate: f_t = sigmoid(W_f @ [h_prev, x_t] + b_f)
    """
    if h_prev.ndim == 1:
        return sigmoid(W_f @ np.concatenate([h_prev, x_t], axis=0) + b_f)
    # 1. Concatenate the previous hidden state and current input
    # Ensure they are joined along the first axis (vertical stack for 1D)
    concat = np.concatenate([h_prev, x_t], axis=1)
    
    # 2. Linear transformation followed by sigmoid activation
    # W_f shape should be (hidden_dim, hidden_dim + input_dim)
    f_t = sigmoid(np.dot(concat, W_f.T) + b_f)
    
    return f_t