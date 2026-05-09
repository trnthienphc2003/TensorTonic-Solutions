import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    # Your implementation here
    B, H, W, C = features.shape
    features = features.reshape((B, -1))

    x1 = np.maximum(features @ W1 + b1, 0)
    # features = np.concatenate([features, x1], axis=-1)
    # assert False, features.shape

    x2 = np.maximum(x1 @ W2 + b2, 0)
    # assert False, x2.shape
    x3 = x2 @ W3 + b3
    
    return x3