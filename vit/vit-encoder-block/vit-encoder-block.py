import numpy as np

def gelu(x: np.ndarray) -> np.ndarray:
    return .5 * x * (1. + np.tanh(np.sqrt(2. / np.pi) * (x + 0.044715 * x ** 3)))

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def norm(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True))

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    # YOUR CODE HERE
    B, N, D = x.shape
    # D == embed_dim
    # Self attention: W{q,k,v}.shape = (D, D)
    # x_norm: (B, N, D)
    x_norm = norm(x)

    # Q: (D, D)
    Q = x_norm @ Wq
    K = x_norm @ Wk
    V = x_norm @ Wv
    
    dk = D // num_heads
    Q = Q.reshape((B, N, num_heads, dk)).transpose((0, 2, 1, 3))
    K = K.reshape((B, N, num_heads, dk)).transpose((0, 2, 1, 3))
    V = V.reshape((B, N, num_heads, dk)).transpose((0, 2, 1, 3))

    
    attn = softmax(Q @ K.transpose((0, 1, 3, 2)) / np.sqrt(num_heads)) @ V
    # assert attn.shape == (B, N, num_heads, dk)
    attn = attn.transpose((0, 2, 1, 3)).reshape((B, N, D))

    # assert False, Wo.shape

    x_new = x + attn @ Wo
    x_new_norm = norm(x_new)

    mlp = gelu(x_new_norm @ W1) @ W2
    output = x_new + mlp
    
    return output