import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Apply 2D max pooling (shape simulation).
    """
    # YOUR CODE HERE
    B, H, W, C = x.shape
    patches = x.transpose((0, 3, 1, 2))
    patches = np.lib.stride_tricks.sliding_window_view(patches, window_shape=(kernel_size, kernel_size), axis=(-2, -1))[:, :, ::stride, ::stride, :, :]
    patches = patches.max(axis=(-2, -1))

    patches = patches.transpose((0, 2, 3, 1))
    return patches

    # assert False, patches.shape
    # return None