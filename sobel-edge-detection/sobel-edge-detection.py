def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    # Write code here
    def _zeros(H, W):
        return [[0 for _ in range(W)] for _ in range(H)]
    H, W = len(image), len(image[0])
    padded_image = _zeros(H + 2, W + 2)
    for i in range(H):
        for j in range(W):
            padded_image[i + 1][j + 1] = image[i][j]

    def _convolve(padded_image, kernel):
        kH, kW = len(kernel), len(kernel[0])
        H, W = len(padded_image) - kH + 1, len(padded_image[0]) - kW + 1
        filtered_im = _zeros(H, W)

        for i in range(H):
            for j in range(W):
                for x in range(kH):
                    for y in range(kW):
                        filtered_im[i][j] += kernel[x][y] * padded_image[i + x][j + y]
        return filtered_im

    K_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    K_y = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]

    G_x, G_y = _convolve(padded_image, K_x), _convolve(padded_image, K_y)
    return [[(G_x[i][j] ** 2 + G_y[i][j] ** 2) ** .5 for j in range(W)] for i in range(H)]