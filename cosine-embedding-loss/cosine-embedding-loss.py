import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    x1, x2 = np.asarray(x1), np.asarray(x2)
    if label == 1:
        return 1. - x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        
    return np.maximum(0., x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)) - margin)
