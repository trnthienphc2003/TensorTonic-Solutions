import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x)
    mean = np.mean(x)
    n = x.shape[0]

    var = np.sum((x - mean) ** 2, axis = 0) / (n - 1)
    return var, np.sqrt(var)