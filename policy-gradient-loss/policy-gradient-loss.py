def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """

    N = len(log_probs)
    assert N == len(rewards)

    G = rewards
    for i in range(N - 2, -1, -1):
        G[i] += gamma * G[i + 1]

    # assert False, G
    G_mean = sum(G) / N
    A = [G[i] - G_mean for i in range(N)]

    loss = sum([log_probs[i] * A[i] for i in range(N)])
    loss = -loss / N

    return loss
    