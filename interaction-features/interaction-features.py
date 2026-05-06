def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    # Write code here
    for idx, feat in enumerate(X):
        n = len(feat)
        nxt = [feat[i] * feat[j] for i in range(n) for j in range(i + 1, n)]
        X[idx].extend(nxt)
    return X