import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    x = np.asarray(x)
    n = x.shape[0]

    # RNG handling (reproducibility)
    if rng is None:
        rng = np.random.default_rng()

    # Sample indices: (n_bootstrap, n)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    # Bootstrap samples
    samples = x[idx]  # shape: (n_bootstrap, n)

    # Compute means
    boot_means = samples.mean(axis=1)

    # Confidence interval (percentile method)
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return boot_means, lower, upper