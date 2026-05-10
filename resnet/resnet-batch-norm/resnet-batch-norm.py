import numpy as np

def batch_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x_hat = (x - x.mean(axis=0, keepdims=True)) / (np.sqrt(x.var(axis=0, keepdims=True)) + eps)
    return gamma * x_hat + beta

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    # YOUR CODE HERE
    x, W1, W2 = np.asarray(x), np.asarray(W1), np.asarray(W2)
    gamma1, beta1 = np.asarray(gamma1), np.asarray(beta1)
    gamma2, beta2 = np.asarray(gamma2), np.asarray(beta2)
    eps = 1e-5
    if mode == "post":
        x1 = (x @ W1)
        x_norm = batch_norm(x1, gamma1, beta1, eps)

        x_relu = np.maximum(x_norm, 0)
        x2 = (x_relu @ W2)

        x2_norm = batch_norm(x2, gamma2, beta2, eps)
        x_skip = x + x2_norm
        out = np.maximum(x_skip, 0)
        return {
            "output": out,
            "mode": mode
        }
    elif mode == "pre":
        x_norm = batch_norm(x, gamma1, beta1, eps)
        x1_relu = np.maximum(x_norm, 0)
        x1 = (x1_relu @ W1)

        x2_norm = batch_norm(x1, gamma2, beta2, eps)
        x2_relu = np.maximum(x2_norm, 0)
        x2 = (x2_relu @ W2)
        out = x + x2

        return {
            "output": out,
            "mode": mode
        }
    else:
        raise ValueError(f"The function expected mode as \"post\" or \"pre\", found \"{mode}\"")
    return None