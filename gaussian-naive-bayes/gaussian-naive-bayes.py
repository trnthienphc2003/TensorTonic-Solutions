import math
import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    # Write code here
    (X_train, y_train, X_test) = map(
        lambda x: np.asarray(x),
        (X_train, y_train, X_test)
    )

    N_train, D = X_train.shape
    C = y_train.max() + 1
    N_test = X_test.shape
    eps = 1e-8

    counts = np.bincount(y_train, minlength=C)
    prob = counts / N_train
    log_prob = np.log(prob)

    sums = np.zeros((C, D))
    np.add.at(sums, y_train, X_train)

    mean = sums / counts[..., None]
    sq_diff = (X_train - mean[y_train]) ** 2
    
    var = np.zeros((C, D), dtype=np.float64)
    np.add.at(var, y_train, sq_diff)
    var /= counts[..., None]
    var += eps

    # log_pos = np.zeros((N_test, C, D))
    diff = X_test[:, None, :] - mean[None, :, :]
    log_pos = (
        - 0.5 * np.log(2. * np.pi * var)
        - diff ** 2 / (2. * var)
    ).sum(axis=-1) + log_prob[None, :]

    return np.argmax(log_pos, axis=-1).tolist()
    # assert False, np.argmax(log_pos, axis=-1)