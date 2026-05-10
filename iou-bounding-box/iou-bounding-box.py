def s(box) -> float:
    return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)

def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    assert len(box_a) == 4 and len(box_b) == 4
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    s_in = s([x1, y1, x2, y2])
    return s_in / (s(box_a) + s(box_b) - s_in)