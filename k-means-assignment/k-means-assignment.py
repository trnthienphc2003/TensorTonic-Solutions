def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    n = len(points)
    m = len(centroids)

    def sq_l2_dist(p1, p2):
        assert len(p1) == len(p2)
        ans = 0
        for i in range(len(p1)):
            d = p2[i] - p1[i]
            ans += d * d
        return ans

    ans = [0] * n
    for i in range(n):
        ans_dist = float("inf")
        for j in range(m):
            d_ij = sq_l2_dist(points[i], centroids[j])
            if d_ij < ans_dist:
                ans_dist = d_ij
                ans[i] = j

    return ans