from itertools import accumulate

def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    returns = [r + 1 for r in returns]
    prod = accumulate(returns, lambda x, y: x * y)
    return [p - 1 for p in prod]