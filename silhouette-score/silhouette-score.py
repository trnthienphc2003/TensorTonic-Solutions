import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here\
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    n = X.shape[0]
    classes, inverse = np.unique(labels, return_inverse=True)
    k = len(classes)

    if k < 2 or k >= n:
        raise ValueError("Silhouette score requires 2 <= n_clusters <= n_samples - 1")

    # Pairwise squared Euclidean distances
    sq_norms = np.sum(X * X, axis=1, keepdims=True)
    D2 = sq_norms + sq_norms.T - 2 * X @ X.T
    D2 = np.maximum(D2, 0.0)

    D = np.sqrt(D2)

    # One-hot cluster membership: shape (n, k)
    M = np.eye(k)[inverse]

    # Cluster sizes: shape (k,)
    counts = M.sum(axis=0)

    # Sum of distances from each point to each cluster: shape (n, k)
    dist_sums = D @ M

    # Mean distance from each point to each cluster
    mean_dists = dist_sums / counts

    # a(i): mean distance to own cluster, excluding itself
    own_counts = counts[inverse]
    own_dist_sums = dist_sums[np.arange(n), inverse]

    a = np.where(
        own_counts > 1,
        own_dist_sums / (own_counts - 1),
        0.0
    )

    # b(i): minimum mean distance to another cluster
    mean_dists[np.arange(n), inverse] = np.inf
    b = np.min(mean_dists, axis=1)

    # silhouette score
    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / denom, 0.0)

    # sklearn convention: singleton clusters get score 0
    s = np.where(own_counts > 1, s, 0.0)

    return np.mean(s)