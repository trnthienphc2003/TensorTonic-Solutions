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

        for j in range(l - 1, r, 1):
            values[j] = values[l - 1] + (j - l + 1) / (r - l + 1) * (values[r] - values[l - 1])

        l = r
    return values