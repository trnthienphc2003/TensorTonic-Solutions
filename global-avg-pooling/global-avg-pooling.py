import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    if type(x) != np.ndarray:
        x = np.asarray(x)

    if x.ndim > 4 or x.ndim < 3:
        raise ValueError(f"Invalid shape for x: {x.shape}")
        return

    return np.average(x, axis=(-2, -1))
        