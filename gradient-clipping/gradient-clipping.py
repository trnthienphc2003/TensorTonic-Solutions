import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g)
    g_norm = np.linalg.norm(g)
    coef = np.minimum(max_norm / g_norm, 1.)
    if max_norm <= 0:
        coef = 1.
    # g *= g_norm
    g = coef * g

    return g