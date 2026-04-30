import copy
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    n, d_in, d_out = len(X), len(W), len(b)
    assert len(X[0]) == d_in
    assert len(W[0]) == d_out

    Y = [copy.copy(b) for x in range(n)]
    for k in range(d_in):
        for i in range(n):
            for j in range(d_out):
                Y[i][j] += X[i][k] * W[k][j]

    return Y