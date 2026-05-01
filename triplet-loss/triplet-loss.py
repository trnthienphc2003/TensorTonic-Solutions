import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor, positive, negative = np.asarray(anchor), np.asarray(positive), np.asarray(negative)
    anchor = np.atleast_2d(anchor)
    positive = np.atleast_2d(positive)
    negative = np.atleast_2d(negative)

    dist_pos = np.linalg.norm(positive - anchor, axis = -1)
    dist_neg = np.linalg.norm(negative - anchor, axis = -1)
    # assert False, f"dist_pos: {dist_pos}, dist_neg: {dist_neg}"
    return np.average(np.maximum(dist_pos * dist_pos - dist_neg * dist_neg + margin, 0))