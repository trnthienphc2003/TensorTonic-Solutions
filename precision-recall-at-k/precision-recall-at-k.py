def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    n, m = len(recommended), len(relevant)
    one_hot = [1 if recommended[i] in relevant else 0 for i in range(n)][:k]
    score = sum(one_hot)
    return [score / k, score / m]