import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    X, y = np.asarray(X), np.asarray(y)

    N = X.shape[0]
    idx = np.arange(N)
    if rng is None:
        rng = np.random

    rng.shuffle(idx)
    stop = N if not drop_last else (N // batch_size) * batch_size
    for i in range(0, stop, batch_size):
        # end = i + batch_size
        samp_idx = idx[i : i + batch_size]
        yield X[samp_idx], y[samp_idx]