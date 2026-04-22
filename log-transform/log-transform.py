from math import log
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    return [log(1. + x) for x in values]