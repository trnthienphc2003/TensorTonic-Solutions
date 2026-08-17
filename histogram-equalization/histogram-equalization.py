from itertools import accumulate

def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    # Write code here
    hist = [0 for _ in range(256)]
    H, W = len(image), len(image[0])
    tot_pix = H * W
    for i in image:
        for pix in i:
            hist[pix] += 1

    cdf = list(accumulate(hist))
    cdf_min = next(x for x in cdf if x != 0)
    # assert False, cdf_min
    if tot_pix == cdf_min:
        return [[0 for _ in range(W)] for _ in range(H)]
    # assert False, cdf_min
    for im in image:
        for j in range(len(im)):
            im[j] = round((cdf[im[j]] - cdf_min) / (tot_pix - cdf_min) * 255.)

    return image