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
    for i in range(0, N, batch_size):
        samp_idx = idx[i : i + batch_size]
        if len(samp_idx) < batch_size and drop_last:
            continue
        yield X[samp_idx], y[samp_idx]