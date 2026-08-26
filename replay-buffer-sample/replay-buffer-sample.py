import numpy as np

def replay_buffer_sample(buffer: list, batch_size: int, seed: int) -> list:
    """
    Returns a deterministic sample of transitions.
    """
    # Write code here
    np.random.seed(seed)
    N = len(buffer)
    idx = np.random.choice(N, size=batch_size, replace=False)

    buffer_choice = np.asarray(buffer).copy()
    return buffer_choice[idx]