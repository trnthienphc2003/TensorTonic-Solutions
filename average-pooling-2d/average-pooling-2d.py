def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    H, W = len(X), len(X[0])
    H_out, W_out = H // pool_size, W // pool_size

    out = [[0 for _ in range(W_out)] for x in range(H_out)]
    for i in range(H_out):
        for j in range(W_out):
            for a in range(pool_size):
                for b in range(pool_size):
                    out[i][j] += X[i * pool_size + a][j * pool_size + b]
            out[i][j] /= (pool_size ** 2)

    return out