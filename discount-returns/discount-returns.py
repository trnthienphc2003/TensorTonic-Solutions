def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    # Write code here
    G = rewards
    T = len(rewards)
    for t in range(T - 2, -1, -1):
        G[t] += gamma * G[t + 1]

    return G