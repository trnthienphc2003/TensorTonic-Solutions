def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.
    """

    n = len(U)
    uv_dot = [U[i] * V[i] for i in range(n)]
    e = r - sum(uv_dot)

    new_U = [U[i] + lr * (e * V[i] - reg * U[i]) for i in range(n)]
    new_V = [V[i] + lr * (e * U[i] - reg * V[i]) for i in range(n)]

    return new_U, new_V