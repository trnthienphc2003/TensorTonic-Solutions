def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    k = len(weights)
    def _dot_product(x, y):
        assert len(x) == len(y)
        return sum(x[i] * y[i] for i in range(len(x)))

    sum_w = sum(weights)
    return [_dot_product(values[i:i+k], weights) / sum_w for i in range(len(values) - k + 1)]