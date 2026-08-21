def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    out = []
    for i in range(period):
        sub_seq = series[i::period]
        out.append(sum(sub_seq) / len(sub_seq))

    return out