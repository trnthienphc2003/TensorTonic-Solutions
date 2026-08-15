def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Your code here
    N = len(tokens)
    words = list(set(tokens))
    counts = {}
    sum = {}
    for i in range(1, N):
        counts[(tokens[i - 1], tokens[i])] = counts.get((tokens[i - 1], tokens[i]), 0) + 1
        sum[tokens[i - 1]] = sum.get(tokens[i - 1], 0) + 1

    probs = {}
    for u in words:
        s_u = sum.get(u, 0)
        for v in words:
            c_uv = counts.get((u, v), 0)
            probs[(u, v)] = (c_uv + 1) / (s_u + len(words))

    return counts, probs