def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    # Write code here

    n = len(series)
    l, b = [0] * n, [0] * n
    l[0] = series[0]
    b[0] = series[1] - series[0]
    for i in range(1, n):
        l[i] = alpha * series[i] + (1 - alpha) * (l[i - 1] + b[i - 1])
        b[i] = beta * (l[i] - l[i - 1]) + (1 - beta) * b[i - 1]
    return l