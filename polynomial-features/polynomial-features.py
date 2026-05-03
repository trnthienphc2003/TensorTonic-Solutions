def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    return [[v ** i for i in range(degree + 1)] for v in values]