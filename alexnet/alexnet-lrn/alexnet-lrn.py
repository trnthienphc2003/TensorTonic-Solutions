import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Apply Local Response Normalization across channels.
    """
    # YOUR CODE HERE
    B, H, W, C = x.shape
    out_denom = np.zeros_like(x)

    sq_x = x ** 2
    for c in range(C):
        l = max(0, c - n // 2)
        r = min(C - 1, c + n // 2)
        out_denom[..., c] = np.sum(sq_x[..., l:r+1], axis=-1)

    out_denom = (k + alpha * out_denom) ** beta
    return x / out_denom