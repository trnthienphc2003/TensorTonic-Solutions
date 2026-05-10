def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    # Write code here
    n = len(boxes)
    assert n == len(scores)

    idx = [i for i in range(n)]
    idx.sort(key = lambda x: scores[x], reverse=True)
    # assert False, idx

    def _s(bbox) -> float:
        return max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0)

    def _iou(bbox_a, bbox_b) -> float:
        assert len(bbox_a) == 4 and len(bbox_b) == 4
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])

        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])

        s_in = _s([x1, y1, x2, y2])
        return s_in / (_s(bbox_a) + _s(bbox_b) - s_in)

    ans = []
    rem = []
    for i, idx_i in enumerate(idx):
        if i not in rem:
            ans.append(idx_i)
            for j in range(i + 1, len(idx)):
                if(_iou(boxes[idx[i]], boxes[idx[j]]) >= iou_threshold):
                    rem.append(j)
    return ans
                    