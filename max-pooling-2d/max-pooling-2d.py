def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    # Write code here
    H, W = len(X), len(X[0])
    H_out, W_out = H // pool_size, W // pool_size

    out = [[-float("inf") for _ in range(W_out)] for x in range(H_out)]
    for i in range(H_out):
        for j in range(W_out):
            for a in range(pool_size):
                for b in range(pool_size):
                    out[i][j] = max(out[i][j], X[i * pool_size + a][j * pool_size + b])

    return out