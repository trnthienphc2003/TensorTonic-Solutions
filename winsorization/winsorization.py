from math import ceil, floor

def winsorize(values: list, lower_pct: float, upper_pct: float) -> list:
    """
    Returns values clipped to the interpolated percentile bounds.
    """
    # Write code here
    n = len(values)
    s = sorted(values)

    def _percentile(arr, p):
        n = len(arr)
        if p <= 0:
            return arr[0]
        if p >= 100:
            return arr[n - 1]

        k = (n - 1) * p / 100
        if ceil(k) >= n:
            return arr[floor(k)]

        return arr[floor(k)] + (k - floor(k)) * (arr[ceil(k)] - arr[floor(k)])

    def _clip(x, l, r):
        x = min(x, r)
        x = max(x, l)
        return x

    lo = _percentile(s, lower_pct)
    hi = _percentile(s, upper_pct)
    return [_clip(x, lo, hi) for x in values]