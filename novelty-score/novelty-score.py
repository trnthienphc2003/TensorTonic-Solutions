def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    from math import log2
    return sum([-log2(item_counts[x] / n_users) for x in recommendations]) / len(recommendations)