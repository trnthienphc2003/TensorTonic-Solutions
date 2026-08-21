def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """

    N = len(series)
    mean = sum(series) / N

    centered_seq = list(map(lambda x: x - mean, series))
    var = sum(x * x for x in centered_seq)

    out = [0 for _ in range(max_lag + 1)]
    for k in range(max_lag + 1):
        sub_a = centered_seq[:N - k]
        sub_b = centered_seq[k:]

        out[k] = sum(sa * sb for sa, sb in zip(sub_a, sub_b)) / var if var > 0 else (1.0 if k == 0 else 0.0)

    return out