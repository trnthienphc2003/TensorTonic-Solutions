import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x)
    values, counts = np.unique(x, return_counts=True)
    mode_value = values[np.argmax(counts)]

    return np.mean(x), np.median(x), mode_value