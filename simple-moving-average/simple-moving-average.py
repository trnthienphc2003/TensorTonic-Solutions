def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    # Write code here
    n = len(values)
    return [sum(values[i:i+window_size]) / window_size for i in range(n - window_size + 1)]