def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    # Write code here
    from math import exp
    center = size // 2
    k = [[exp(-((x - center) ** 2 + (y - center) ** 2) / (2. * sigma ** 2)) for y in range(size)] for x in range(size)]

    s = sum([sum(x) for x in k])
    norm_k = [[x / s for x in i] for i in k]
    return norm_k