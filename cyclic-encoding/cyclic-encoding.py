from math import pi, sin, cos
def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    return [[sin(2. * pi * v / period), cos(2. * pi * v / period)] for v in values]