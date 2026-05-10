import numpy as np

def norm(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True))

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    # YOUR CODE HERE

    B, N, D = encoder_output.shape
    
    h_cls = encoder_output[:, 0, :]
    h_norm = norm(h_cls)

    if W_head is None:
        W_head = np.random.randn(D, num_classes) * 0.02

    logits = h_norm @ W_head
    return logits