import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    x, Weight, b = np.asarray(x), np.asarray(W), np.asarray(b)
    while x.ndim == 3:
        x = np.expand_dims(x, axis=0)

    N, C_in, H, W = x.shape
    # assert False, W
    C_out, _, KH, KW = Weight.shape

    patches = np.lib.stride_tricks.sliding_window_view(
        x, (KH, KW), axis=(2, 3)
    )

    # assert False, f" x: {x} \n patches: {patches.shape}"

    patches = patches[:, :, ::1, ::1, :, :]
    

    H_out = H - KH + 1
    W_out = W - KW + 1
    ans = np.einsum("nchwkl,ockl->nohw", patches, Weight)
    ans += b[None, :, None, None]
    return ans
    