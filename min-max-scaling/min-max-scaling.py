def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here 
    n = len(data[0])
    m = len(data)
    mn = [float("inf") for _ in range(n)]
    mx = [-float("inf") for _ in range(n)]
    for x in data:
        for j, y in enumerate(x):
            mn[j] = min(mn[j], y)
            mx[j] = max(mx[j], y)

    for i in range(m):
        for j in range(n):
            if mn[j] == mx[j]:
                data[i][j] = 0
            else:
                data[i][j] = (data[i][j] - mn[j]) / (mx[j] - mn[j])

    return data