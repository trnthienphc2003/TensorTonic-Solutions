import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet loss using squared Euclidean distance.

    Supports:
    - (D,)
    - (N, D)
    """
    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)

    anchor = np.atleast_2d(anchor)
    positive = np.atleast_2d(positive)
    negative = np.atleast_2d(negative)

    if not (anchor.shape == positive.shape == negative.shape):
        raise ValueError("anchor, positive, negative must have same shape")

    # Squared L2 distances (no sqrt → faster + standard)
    dist_pos = np.sum((anchor - positive) ** 2, axis=-1)
    dist_neg = np.sum((anchor - negative) ** 2, axis=-1)

    loss = np.maximum(dist_pos - dist_neg + margin, 0.0)

    return loss.mean()