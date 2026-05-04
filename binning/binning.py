def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    w = (max(values) - min(values)) / num_bins
    if(w == 0):
        return [0] * len(values)
    return [min(int((values[i] - min(values)) / w), num_bins - 1) for i in range(len(values))]