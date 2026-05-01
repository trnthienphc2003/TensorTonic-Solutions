import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply a 4x4 homogeneous transform to 3D point(s).

    Parameters
    ----------
    T : array-like, shape (4, 4)
        Homogeneous transformation matrix.
    points : array-like, shape (3,) or (..., 3)
        One point or a batch/grid of 3D points.

    Returns
    -------
    transformed : ndarray, shape (3,) or (..., 3)
        Transformed 3D point(s), preserving input point shape.
    """
    T = np.asarray(T, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)

    if T.shape != (4, 4):
        raise ValueError(f"T must have shape (4, 4), got {T.shape}")

    if points.shape[-1] != 3:
        raise ValueError(f"points must have last dimension 3, got {points.shape}")

    R = T[:3, :3]
    t = T[:3, 3]

    return points @ R.T + t