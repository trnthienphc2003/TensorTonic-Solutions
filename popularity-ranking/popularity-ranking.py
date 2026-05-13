def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    return [i[1] / (i[1] + min_votes) * i[0] + min_votes / (i[1] + min_votes) * global_mean for i in items]