import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    # pass
    C = np.asarray(C)
    # assert C.ndim == 2 and C.shape[0] == C.shape[1]

    val_R, val_C = np.sum(C, axis=1, keepdims=True), np.sum(C, axis=0, keepdims=True)
    N = np.sum(C)

    E = 1. / N * (val_R @ val_C)
    chi2 = np.sum(np.pow(C - E, 2) / E)
    return chi2, E