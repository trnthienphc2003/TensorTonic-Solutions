import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.asarray(points)
    ori_1d = False
    if points.ndim == 1:
        points = np.atleast_2d(points)
        ori_1d = True

    n_pts = points.shape[0]
    # assert False, points.shape
    points = np.concatenate((points, np.ones((n_pts, 1))), axis=-1)
    # assert False, points
    trans_pts = (points @ np.asarray(T).T)[..., :3]
    if ori_1d:
        trans_pts = np.squeeze(trans_pts)

    return trans_pts