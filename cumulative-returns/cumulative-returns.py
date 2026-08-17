from itertools import accumulate

def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    out = []
    cum = 1.
    for r in returns:
        cum *= (1. + r)
        out.append(cum - 1)
    return out