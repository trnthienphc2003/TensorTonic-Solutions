import numpy as np

def compute_advantage(states, rewards, V, gamma):
    states = np.asarray(states, dtype=np.int32)
    rewards = np.asarray(rewards, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    if gamma == 0:
        G = rewards
    else:
        N = len(rewards)
        discounts = gamma ** np.arange(N)
        G = np.cumsum((rewards * discounts)[::-1])[::-1] / discounts

    return G - V[states]