def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    # Write code here
    out = []
    n = len(values)
    for i in range(n - window_size + 1):
        sub_seq = values[i:i+window_size]
        mu = sum(sub_seq) / window_size
        sigma = (sum((x - mu) ** 2 for x in sub_seq) / window_size) ** .5
        out.append(sigma)

    return out