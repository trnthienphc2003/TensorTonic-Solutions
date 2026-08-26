import numpy as np

def q_learning_update(Q: list, s: int, a: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as Q.
    """
    # Write code here
    Q_new = np.asarray(Q, dtype=np.float64).copy()
    y = r + gamma * np.max(Q_new[s_next])

    Q_new[s, a] += alpha * (y - Q_new[s, a])
    return Q_new