import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    x = np.transpose(x, (0, 3, 1, 2))
    x_window = np.lib.stride_tricks.sliding_window_view(x, (2, 2), axis=(2, 3))
    x_window = x_window[:, :, ::2, ::2, :, :]
    # assert False, x_window.shape
    x_window = np.max(x_window, axis=(-1, -2))
    x_window = np.transpose(x_window, (0, 2, 3, 1))
    return x_window