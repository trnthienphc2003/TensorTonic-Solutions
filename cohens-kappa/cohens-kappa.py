import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here

    N = len(rater1)
    rater1, rater2 = np.asarray(rater1), np.asarray(rater2)
    p0 = np.average(rater1 == rater2)

    label = max(rater1.max().item(), rater2.max().item()) + 1
    # assert False, label

    cnt1, cnt2 = np.bincount(rater1, minlength=label), np.bincount(rater2, minlength=label)
    pe = np.dot(cnt1, cnt2) / (N ** 2)
    # assert False, p0

    if abs(pe - 1.0) < 1e-8:
        return 1.0
    return (p0 - pe) / (1. - pe)
    