def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    L = (6. / fan_in) ** .5
    n, m = len(W), len(W[0])
    for i in range(n):
        for j in range(m):
            W[i][j] = 2. * W[i][j] * L - L

    return W