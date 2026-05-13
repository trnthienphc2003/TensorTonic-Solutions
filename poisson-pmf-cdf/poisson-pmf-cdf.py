import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    idx = np.asarray([1])
    idx = np.append(idx, np.arange(1, k + 1))
    # assert False, idx
    pmf = np.exp(-lam) * lam ** (np.arange(k + 1)) / np.cumprod(idx, axis=-1)
    cdf = np.cumsum(pmf, -1)
    # assert False, np.__version__
    return pmf[-1], cdf[-1]