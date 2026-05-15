import numpy as np

def gelu(x: np.ndarray) -> np.ndarray:
    return .5 * x * (1. + np.tanh(np.sqrt(2. / np.pi) * (x + 0.044715 * x ** 3)))

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)

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

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int, pos_embed: np.ndarray = None) -> np.ndarray:
    """
    Add position embeddings to patch embeddings.
    pos_embed: position embedding of shape (1, N, D). If None, initialize randomly.
    """
    # YOUR CODE HERE
    B, N, D = patches.shape
    if pos_embed is None:
        pos_embed = np.random.randn((1, N, D)) * 0.02

    return patches + pos_embed

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    cls_token: shape (1, 1, D). If None, initialize randomly.
    """
    # YOUR CODE HERE
    B, N, D = patches.shape
    if cls_token is None:
        cls_token = np.random.randn((1, 1, D)) * 0.01

    patches = np.concatenate([np.broadcast_to(cls_token, (B, 1, D)), patches], axis=1)
    return patches

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

def vit_encoder_block(x, embed_dim, num_heads, mlp_ratio, Wq, Wk, Wv, Wo, W1, W2):
    B, N, D = x.shape
    dk = D // num_heads
    x_norm = norm(x)

    # Project and Split Heads
    Q = (x_norm @ Wq).reshape(B, N, num_heads, dk).transpose(0, 2, 1, 3)
    K = (x_norm @ Wk).reshape(B, N, num_heads, dk).transpose(0, 2, 1, 3)
    V = (x_norm @ Wv).reshape(B, N, num_heads, dk).transpose(0, 2, 1, 3)

    # Scaled Dot-Product Attention (Fix: use dk, not num_heads)
    scaled_dot = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dk)
    attn = softmax(scaled_dot) @ V 
    
    # Re-combine heads and apply output projection
    attn = attn.transpose(0, 2, 1, 3).reshape(B, N, D)
    x = x + (attn @ Wo) # Residual 1

    # MLP Block
    x_norm2 = norm(x)
    mlp_out = gelu(x_norm2 @ W1) @ W2
    x = x + mlp_out # Residual 2
    
    return x

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        # Initialize weights here
        if W_patch is None:
            W_patch = np.random.randn((patch_size * patch_size * C, embed_dim)) * 0.02
        self.W_patch = W_patch

        if cls_token is None:
            cls_token = np.random.randn((1, 1, embed_dim)) * 0.02
        self.cls_token = cls_token

        if pos_embed is None:
            pos_embed = np.random.randn((1, num_patches + 1, embed_dim)) * 0.02
        self.pos_embed = pos_embed

        if encoder_weights is None:
            encoder_weights = []
            hidden_dim = int(mlp_ratio * embed_dim)
            for i in range(depth):
                encoder_weights.append({
                    "Wq": np.random.randn((embed_dim, embed_dim)) * 0.02,
                    "Wk": np.random.randn((embed_dim, embed_dim)) * 0.02,
                    "Wv": np.random.randn((embed_dim, embed_dim)) * 0.02,
                    
                    "Wo": np.random.randn((embed_dim, embed_dim)) * 0.02,
                    "W1": np.random.randn((embed_dim, hidden_dim)) * 0.02,
                    "W2": np.random.randn((hidden_dim, embed_dim)) * 0.02,
                })
        self.encoder_weights = encoder_weights

        if W_head is None:
            W_head = np.random.randn((embed_dim, num_classes)) * 0.02
        self.W_head = W_head
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        # YOUR CODE HERE
        x = patch_embed(x, self.patch_size, self.embed_dim, self.W_patch)

        # CLS
        x = prepend_class_token(x, self.embed_dim, self.cls_token)

        # Position embedding
        x = add_position_embedding(x, self.num_patches, self.embed_dim, self.pos_embed)

        # L x transformer block
        for block in self.encoder_weights:
            Wq, Wk, Wv = block["Wq"], block["Wk"], block["Wv"]
            Wo, W1, W2 = block["Wo"], block["W1"], block["W2"]

            x = vit_encoder_block(x, self.embed_dim, self.num_heads, self.mlp_ratio, Wq, Wk, Wv, Wo, W1, W2)

        # Classification
        output = classification_head(x, self.num_heads, self.W_head)
        return output