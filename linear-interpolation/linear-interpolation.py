def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    l = 0
    while l < len(values):
        if values[l] is not None:
            l += 1
            continue

        r = l + 1
        while r < len(values) and values[r] is None:
            r += 1

        step = (values[r] - values[l - 1]) / (r - l + 1)
        for j in range(l, r, 1):
            values[j] = values[j - 1] + step

        l = r
    return values