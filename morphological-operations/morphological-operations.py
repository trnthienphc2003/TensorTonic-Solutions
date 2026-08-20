def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    # Write code here
    H, W = len(image), len(image[0])
    kH, kW = len(kernel), len(kernel[0])
    centre_kY, centre_kX = kH // 2, kW // 2
    out = [[0 for _ in range(W)] for _ in range(H)]

    def _in_bound_image(y, x):
        return y >= 0 and y < H and x >= 0 and x < W

    flag = False
    if operation == 'dilate':
        for i in range(H):
            for j in range(W):
                flag = False
                
                for dy in range(- (kH // 2), (kH // 2) + 1):
                    for dx in range(- (kW // 2), (kW // 2) + 1):
                        if kernel[centre_kY + dy][centre_kX + dx] != 1:
                            continue
                        if _in_bound_image(i + dy, j + dx):
                            flag |= image[i + dy][j + dx] == 1
                            if flag:
                                break
                    if flag:
                        break

                if flag:
                    out[i][j] = 1

    else:
        for i in range(H):
            for j in range(W):
                flag = True
                
                for dy in range(- (kH // 2), (kH // 2) + 1):
                    for dx in range(- (kW // 2), (kW // 2) + 1):
                        if kernel[centre_kY + dy][centre_kX + dx] != 1:
                            continue
                            
                        if _in_bound_image(i + dy, j + dx):
                            flag &= image[i + dy][j + dx] == 1
                        else:
                            flag = False
                        if not flag:
                            break
                        
                    if not flag:
                        break

                if flag:
                    out[i][j] = 1

    return out