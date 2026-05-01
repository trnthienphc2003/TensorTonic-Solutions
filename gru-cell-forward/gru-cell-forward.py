import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def _tanh(x):
    return 1. / (_sigmoid(-x) + 1)

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]

    D, H = len(Wh), len(Uh)

    x, conv_x = _as2d(x, D)
    h_prev, conv_h = _as2d(h_prev, H)

    # assert False, Wz.shape
    zt = _sigmoid(x @ Wz + h_prev @ Uz + bz)
    # assert False
    rt = _sigmoid(x @ Wr + h_prev @ Ur + br)
    cand_h = np.tanh(x @ Wh + (rt * h_prev) @ Uh + bh)
    h = (1 - zt) * h_prev + zt * cand_h

    if conv_h:
        h = h.squeeze(0)
    return h