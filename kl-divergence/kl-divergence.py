import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p, q = np.array(p), np.array(q)
    return p.dot(np.log(p)) - p.dot(np.log(q + eps))