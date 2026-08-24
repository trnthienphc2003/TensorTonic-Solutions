def top_k_recommendations(scores: list, rated_indices: list, k: int) -> list:
    """
    Returns the highest-scoring unrated item indices.
    """
    # Write code here
    n = len(scores)
    filtered = [(i, scores[i]) for i in range(n) if i not in rated_indices]
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:k]
    idx, _ = zip(*filtered)

    return list(idx)
    