import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    idx = np.arange(k + 1)
    pos, inv = np.pow(p, idx), np.pow(1. - p, n - idx)
    pmf = (pos * inv) * (comb(n, idx))
    cdf = np.cumsum(pmf)
    return pmf[-1], cdf[-1]