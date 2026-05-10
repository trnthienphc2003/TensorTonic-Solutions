def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    H, W = len(X), len(X[0])
    H_out, W_out = (H - pool_size) // stride + 1, (W - pool_size) // stride + 1

    out = [[-float("inf") for _ in range(W_out)] for x in range(H_out)]
    for i in range(H_out):
        for j in range(W_out):
            for a in range(pool_size):
                for b in range(pool_size):
                    out[i][j] = max(out[i][j], X[i * stride + a][j * stride + b])

    return out