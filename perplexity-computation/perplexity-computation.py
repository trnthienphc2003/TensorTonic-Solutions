def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    prob = [prob_distributions[i][actual_tokens[i]] for i in range(len(actual_tokens))]

    import math
    entropy = -sum([math.log(p) for p in prob]) / len(prob)
    return math.exp(entropy)