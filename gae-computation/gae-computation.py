def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """

    n = len(rewards)
    delta = [rewards[i] + gamma * values[i + 1] - values[i] for i in range(n)]
    # assert False, delta
    A = delta
    for i in range(n - 2, -1, -1):
        A[i] = delta[i] + gamma * lam * A[i + 1]
    return A