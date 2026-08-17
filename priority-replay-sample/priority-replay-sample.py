def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    pow_p = list(map(lambda p: p ** alpha, priorities))
    sum_pow_p = sum(pow_p)
    probs = list(map(lambda p: p / sum_pow_p, pow_p))
    # assert False, probs

    N = len(priorities)
    weights = list(map(lambda p: (N * p) ** (-beta), probs))
    mx_weights = max(weights)
    norm_w = list(map(lambda w: w / mx_weights, weights))
    
    return [probs, norm_w]