import numpy as np

def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    L = (6. / (fan_in + fan_out)) ** .5
    W = np.asarray(W)
    W = W * 2. * L - L
    return W