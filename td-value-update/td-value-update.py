import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    # Write code here
    delta = r + gamma * V[s_next] - V[s]
    V[s] += alpha * delta
    return V