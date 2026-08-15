import numpy as np

def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    words, ids = np.unique(tokens, return_inverse=True)
    V = len(words)
    counts_mat = np.zeros((V, V), dtype=int)

    np.add.at(
        counts_mat,
        (ids[:-1], ids[1:]),
        1
    )

    prefix_counts = counts_mat.sum(axis=1)
    probs_mat = (counts_mat + 1) / (prefix_counts[:, None] + V)

    counts = {
        (words[i], words[j]): counts_mat[i, j].item()
        for i in range(V)
        for j in range(V)
        if counts_mat[i, j] > 0
    }

    probs = {
        (words[i], words[j]): probs_mat[i, j].item()
        for i in range(V)
        for j in range(V)
    }

    return counts, probs