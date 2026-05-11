def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    # Write code here
    n = len(similarities)
    pos_s = [max(similarities[i], 0) for i in range(n)]

    sum_s = sum(pos_s)
    if sum_s == 0.:
        return 0
    return sum([pos_s[i] * ratings[i] for i in range(n)]) / sum_s