import numpy as np

def precision_recall_at_k(recommended, relevant, k):
    recommended = np.asarray(recommended)
    relevant = np.asarray(relevant)

    # Take top-k
    recommended_k = recommended[:k]

    # Vectorized membership check
    hits = np.isin(recommended_k, relevant)

    score = hits.sum()

    precision = score / k
    recall = score / len(relevant) if len(relevant) > 0 else 0.0

    return [precision, recall]