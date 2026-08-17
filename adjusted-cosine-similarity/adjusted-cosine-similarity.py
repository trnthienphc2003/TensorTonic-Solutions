# from functools import reduce

def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    
    def _column(X, idx):
        assert len(X) != 0 and len(X[0]) > idx
        return list(map(lambda x: x[idx], X))
    
    def _l2(v):
        return sum(map(lambda x: x ** 2, v)) ** .5

    def _dot_prod(u, v):
        return sum(u[i] * v[i] for i in range(len(u)))

    def _avg(v):
        filtered_v = list(filter(lambda x: x != 0, v))
        assert len(filtered_v) != 0
        return sum(filtered_v) / len(filtered_v)
        
    valid_rating = list(filter(lambda x: x[item_i] != 0 and x[item_j] != 0, ratings_matrix))
    if len(valid_rating) == 0:
        return 0.0
    mean_u = list(map(lambda x: _avg(x), valid_rating))

    r_item_i, r_item_j = _column(valid_rating, item_i), _column(valid_rating, item_j)
    centered_i = list(r_i - mu for r_i, mu in zip(r_item_i, mean_u))
    centered_j = list(r_j - mu for r_j, mu in zip(r_item_j, mean_u))
    # assert False, valid_rating

    norm_i, norm_j = _l2(centered_i), _l2(centered_j)
    if norm_i == 0.0 or norm_j == 0.0:
        return 0.0
    return _dot_prod(centered_i, centered_j) / (norm_i * norm_j)