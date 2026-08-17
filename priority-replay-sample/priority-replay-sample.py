def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    pow_p = [p ** alpha for p in priorities]
    sum_pow_p = sum(pow_p)
    probs = [p / sum_pow_p for p in pow_p]
    # assert False, probs

    N = len(priorities)
    weights = [1. / (N * p) ** beta for p in probs]
    mx_weights = max(weights)
    norm_w = [w / mx_weights for w in weights]
    
    return [probs, norm_w]