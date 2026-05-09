import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    # YOUR CODE HERE
    B, H, W, C = image.shape
    patches = image.reshape((B, H // patch_size, patch_size, W // patch_size, patch_size, C))
    patches = patches.transpose((0, 1, 3, 2, 4, 5))
    # assert False, patches.shape
    patches = patches.reshape((B, -1, patch_size * patch_size * C))
    if W_proj is None:
        W_proj = np.random.randn((patch_size * patch_size * C, embed_dim)) * 0.02
    embeddings = patches @ W_proj

    return embeddings