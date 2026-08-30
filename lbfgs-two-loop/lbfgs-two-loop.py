def _dot_prod(a, b):
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad: list, s_list: list, y_list: list) -> list:
    """
    Returns the L-BFGS descent direction from the stored history.
    """
    # Write code here
    N = len(grad)
    M = len(s_list)
    q = grad[:]

    alphas = [0.0 for _ in range(M)]
    rhos = [0.0 for _ in range(M)]
    for i in range(M - 1, -1, -1):
        rhos[i] = 1. / _dot_prod(y_list[i], s_list[i])
        alphas[i] = rhos[i] * _dot(s_list[i], q)
        q = [q[j] - alphas[i] * y_list[i][j] for j in range(N)]

    gamma = _dot(s_list[-1], y_list[-1]) / _dot(y_list[-1], y_list[-1])
    r = [gamma * x for x in q]

    betas = [0.0 for _ in range(M)]
    for i in range(M):
        betas[i] = rhos[i] * _dot_prod(y_list[i], r)
        r = [r[j] + s_list[i][j] * (alphas[i] - betas[i]) for j in range(N)]

    return [-r[i] for i in range(N)]