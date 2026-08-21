def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    n = len(values)
    out = []
    for i in range(n - window_size + 1):
        sub_v = values[i:i+window_size]
        sub_v = sorted(sub_v)
        out.append(sub_v[window_size // 2] if window_size % 2 == 1 else (sub_v[(window_size - 1) // 2] + sub_v[(window_size + 1) // 2]) / 2)

    return out