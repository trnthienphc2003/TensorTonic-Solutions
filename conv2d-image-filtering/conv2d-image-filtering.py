def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    N, M = len(image), len(image[0])
    new_image = [[0 for x in range(M + 2 * padding)] for y in range(N + 2 * padding)]

    # assert False, len(new_image[0])
    for i in range(N):
        new_image[i + padding] = [
            image[i][j - padding] if padding <= j < padding + M else 0
            for j in range(M + 2 * padding)
        ]

    KH, KW = len(kernel), len(kernel[0])
    out_H, out_W = ((N + 2 * padding - KH) // stride) + 1, ((M + 2 * padding - KW) // stride) + 1

    ans = [[0 for x in range(out_W)] for y in range(out_H)]
    for i in range(out_H):
        for j in range(out_W):
            for kx in range(KH):
                for ky in range(KW):
                    ans[i][j] += new_image[i * stride + kx][j * stride + ky] * kernel[kx][ky]


    return ans