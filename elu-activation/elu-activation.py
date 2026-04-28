from math import exp
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    return [el if el > 0 else alpha * (exp(el) - 1) for el in x]