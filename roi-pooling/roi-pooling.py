def roi_pool(feature_map, rois, output_size):
    res = []

    for x1, y1, x2, y2 in rois:
        roi_h = y2 - y1
        roi_w = x2 - x1

        roi_max = max(
            feature_map[h][w]
            for h in range(y1, y2)
            for w in range(x1, x2)
        )

        pooled = [[roi_max for _ in range(output_size)] for _ in range(output_size)]

        for i in range(output_size):
            h_start = y1 + (i * roi_h) // output_size
            h_end   = y1 + ((i + 1) * roi_h) // output_size

            for j in range(output_size):
                w_start = x1 + (j * roi_w) // output_size
                w_end   = x1 + ((j + 1) * roi_w) // output_size

                if h_start < h_end and w_start < w_end:
                    pooled[i][j] = max(
                        feature_map[h][w]
                        for h in range(h_start, h_end)
                        for w in range(w_start, w_end)
                    )

        res.append(pooled)

    return res