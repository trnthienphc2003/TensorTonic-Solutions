from math import sin, cos, pi

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    # Write code here
    angle_rad = pi * angle_degrees / 180.
    H, W = len(image), len(image[0])
    out = [[0 for _ in range(W)] for _ in range(H)]
    cy, cx = (H - 1) / 2, (W - 1) / 2

    for y in range(H):
        dy = y - cy
        for x in range(W):
            dx = x - cx
            src_y = round(cy + dy * cos(angle_rad) + dx * sin(angle_rad))
            src_x = round(cx - dy * sin(angle_rad) + dx * cos(angle_rad))
            if src_y >= 0 and src_y < H and src_x >= 0 and src_x < W:
                out[y][x] = image[src_y][src_x]
    return out