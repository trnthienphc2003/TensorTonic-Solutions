import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x)

    n = x.shape[0]
    mean, s = np.mean(x), np.std(x, ddof=1)
    t = (mean - mu0) / (s / np.sqrt(n))

    # assert False, t
    return t