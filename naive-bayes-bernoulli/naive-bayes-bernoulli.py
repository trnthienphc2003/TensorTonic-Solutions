import numpy as np

def naive_bayes_bernoulli(X_train: list, y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a NumPy array of log posteriors.
    """
    # Write code here
    (
        X_train,
        y_train,
        X_test
    ) = map(
        lambda x: np.asarray(x, dtype=np.float64),
        (
            X_train,
            y_train,
            X_test
        )
    )

    classes, cnt = np.unique(y_train, return_counts=True)
    n_class = len(classes)
    # assert False, classes
    n_train, n_feature = X_train.shape
    theta = np.zeros((n_class, n_feature))
    x, y = np.where(y_train[..., None] == classes[None, ...])
    # assert False, y
    np.add.at(
        theta,
        y,
        X_train
    )

    # assert False, theta

    # theta = theta.T #(D, n_class)
    theta = (theta + 1) / (cnt[..., None] + 2)
    # assert False
    theta = theta.T

    p = cnt / n_train

    # (N_test, D)
    cross_entropy = X_test @ np.log(theta) + (1. - X_test) @ np.log(1. - theta)
    prob = np.log(p) + cross_entropy
    return np.round(prob, 4)