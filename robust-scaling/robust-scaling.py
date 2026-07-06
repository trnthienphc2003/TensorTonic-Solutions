def robust_scaling(values):
    """
    Scale values using median and interquartile range:
        scaled = (x - median) / IQR
    """
    if len(values) == 0:
        return []

    def median(a):
        a = sorted(a)
        n = len(a)

        if n == 0:
            raise ValueError("Cannot compute median of empty list")

        mid = n // 2
        if n % 2 == 0:
            return (a[mid - 1] + a[mid]) / 2
        return a[mid]

    v_sorted = sorted(values)
    n = len(v_sorted)
    mid = n // 2

    m = median(v_sorted)

    if n == 1:
        return [0 for _ in values]

    if n % 2 == 0:
        lower = v_sorted[:mid]
        upper = v_sorted[mid:]
    else:
        lower = v_sorted[:mid]
        upper = v_sorted[mid + 1:]

    q1 = median(lower)
    q3 = median(upper)

    iqr = q3 - q1

    if iqr == 0:
        return [0 for _ in values]

    return [(v - m) / iqr for v in values]