from math import floor

def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    # Write code here

    out = [[0. for _ in range(new_w)] for _ in range(new_h)]

    H, W = len(image), len(image[0])
    for i in range(new_h):
        src_y = i * (H - 1) / (new_h - 1) if new_h > 1 else 0
        y0 = floor(src_y)
        y1 = min(y0 + 1, H - 1)
        dy = src_y - y0
        
        for j in range(new_w):
            src_x = j * (W - 1) / (new_w - 1) if new_w > 1 else 0
            x0 = floor(src_x)
            x1 = min(x0 + 1, W - 1)
            dx = src_x - x0

            out[i][j] = (
                image[y0][x0] * (1. - dy) * (1. - dx)
                + image[y1][x0] * dy * (1. - dx)
                + image[y0][x1] * (1. - dy) * dx
                + image[y1][x1] * dy * dx
            )

    return out