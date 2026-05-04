def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    n = len(recommendations)
    # v =[(ground_truth[i][0] in recommendations[i][:k]) for i in range(n)]
    # assert False, v
    return 1. * sum([(ground_truth[i][0] in recommendations[i][:k]) for i in range(n)]) / n