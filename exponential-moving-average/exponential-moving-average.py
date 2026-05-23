def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    # Write code here

    n = len(values)
    EMA = values
    for i in range(1, n):
        EMA[i] *= alpha
        EMA[i] += (1. - alpha) * EMA[i - 1]

    return EMA