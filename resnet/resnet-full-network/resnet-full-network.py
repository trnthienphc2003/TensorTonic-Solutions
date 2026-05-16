import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def identity_block(x, W1, W2):
    # Main path: Conv -> ReLU -> Conv
    y = relu(x @ W1)
    y = y @ W2
    # Shortcut addition then final ReLU
    return relu(y + x)

def projection_block(x, W1, W2, Ws):
    # Main path
    y = relu(x @ W1)
    y = y @ W2
    # Projection path
    shortcut = x @ Ws
    # Addition then final ReLU
    return relu(y + shortcut)

# def conv_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, Ws: np.ndarray) -> np.ndarray:
#     """
#     Forward pass with projection shortcut.
#     """
#     # YOUR CODE HERE
#     y = relu(x @ W1)
#     y = relu(y @ W2)

#     y2 = x @ Ws
#     return y + y2

def conv_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, Ws: np.ndarray) -> np.ndarray:
    x, W1, W2, Ws = map(np.asarray, [x, W1, W2, Ws])

    y = relu(x @ W1)
    y = y @ W2

    shortcut = x @ Ws
    return relu(y + shortcut)

def bottleneck_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray, Ws: np.ndarray = None) -> np.ndarray:
    """
    Bottleneck forward: compress -> process -> expand + skip
    """
    # YOUR CODE HERE
    x0 = x
    
    # Compress
    x = relu(x @ W1)
    # Process
    x = relu(x @ W2)

    # Decompress
    x = x @ W3

    skip = x0 @ Ws if Ws is not None else x0
    x = relu(x + skip)
    return x

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    # YOUR CODE HERE
    x, W1, W2 = np.asarray(x), np.asarray(W1), np.asarray(W2)
    gamma1, beta1 = np.asarray(gamma1), np.asarray(beta1)
    gamma2, beta2 = np.asarray(gamma2), np.asarray(beta2)
    eps = 1e-5
    if mode == "post":
        x1 = (x @ W1)
        x_norm = batch_norm(x1, gamma1, beta1, eps)

        x_relu = np.maximum(x_norm, 0)
        x2 = (x_relu @ W2)

        x2_norm = batch_norm(x2, gamma2, beta2, eps)
        x_skip = x + x2_norm
        out = np.maximum(x_skip, 0)
        return {
            "output": out,
            "mode": mode
        }
    elif mode == "pre":
        x_norm = batch_norm(x, gamma1, beta1, eps)
        x1_relu = np.maximum(x_norm, 0)
        x1 = (x1_relu @ W1)

        x2_norm = batch_norm(x1, gamma2, beta2, eps)
        x2_relu = np.maximum(x2_norm, 0)
        x2 = (x2_relu @ W2)
        out = x + x2

        return {
            "output": out,
            "mode": mode
        }
    else:
        raise ValueError(f"The function expected mode as \"post\" or \"pre\", found \"{mode}\"")
    return None

def resnet_forward(x: np.ndarray, conv1: np.ndarray, W1_b1: np.ndarray, W2_b1: np.ndarray, W1_b2: np.ndarray, W2_b2: np.ndarray, Ws_b2: np.ndarray, fc: np.ndarray):
    """
    Simplified ResNet-18 style forward pass.
    
    Args:
        x: Input tensor (batch, features)
        conv1: Initial weight matrix for input
        W1_b1, W2_b1: Weights for the first identity block
        W1_b2, W2_b2, Ws_b2: Weights for the second convolutional (projection) block
        fc: Final fully connected layer weights
        
    Returns:
        np.ndarray: Classification logits
    """
    params = [x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc]
    x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc = [np.asarray(p) for p in params]
    
    # 1. Initial Processing (Stem)
    # Using ReLU on the initial projection
    x = relu(x @ conv1)
    
    # 2. Block 1: Identity Block
    # y = ReLU(W2 @ ReLU(W1 @ x)) + x
    x = identity_block(x, W1_b1, W2_b1)
    
    # 3. Block 2: Convolutional Block (with projection shortcut Ws)
    # y = ReLU(W2 @ ReLU(W1 @ x)) + (Ws @ x)
    x = conv_block(x, W1_b2, W2_b2, Ws_b2)
    
    # 4. Global Average Pooling (Simulated for 2D/Flat feature vectors)
    # If x is (batch, height, width, channels), we would mean over H/W.
    # Given the @ operator used in your blocks, we assume x is already (batch, features).
    
    # 5. Output Layer (Logits)
    logits = x @ fc
    
    return logits