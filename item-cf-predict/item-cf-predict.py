def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    # Write code here
    n = len(item_similarities)
    filtered_idx = [i for i in range(n) if i != target and user_ratings[i] != 0 and item_similarities[i] > 0]
    if len(filtered_idx) == 0:
        return 0.0
    
    return sum(user_ratings[idx] * item_similarities[idx] for idx in filtered_idx) / sum(item_similarities[idx] for idx in filtered_idx)