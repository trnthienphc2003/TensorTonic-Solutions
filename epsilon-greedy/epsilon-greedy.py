import numpy as np

def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    # Write code here
    rng = np.random.default_rng(seed)
    u = rng.random()
    # assert False, u
    if u < epsilon:
        # assert False, rng.integers(0, len(q_values), 1)[0]
        return int(rng.integers(len(q_values)))
    # return np.argmax(np.asarray(q_values))
    return int(np.argmax(np.asarray(q_values)))