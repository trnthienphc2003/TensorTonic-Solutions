import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, conv_weights: list, conv_biases: list) -> np.ndarray:
    """
    Returns: np.ndarray feature tensor after applying conv layers and max pooling
    """
    # Your implementation here
    # conv_weights, conv_biases = np.asarray(conv_weights), np.asarray(conv_biases)
    B, H, W, C = x.shape
    # KH, KW = conv_weights.shape

    # conv_weights = np.expand_dims(conv_weights, axis=0)
    idx = 0
    for c in config:
        if c == 'M':
            x = maxpool_2x2(x)
        else:
            x = x @ np.asarray(conv_weights[idx]) + np.asarray(conv_biases[idx])
            x = np.maximum(x, 0)
            idx += 1
            # x_window = np.transpose(x, (0, 3, 1, 2))
            # x_window = np.lib.stride_tricks.sliding_window_view(x, (KH, KW), axis=(2, 3))
            # x_window = x_window[:, :, ::KH, ::KW, :, :]
            # x_window = np.einsum("nc")
    return x