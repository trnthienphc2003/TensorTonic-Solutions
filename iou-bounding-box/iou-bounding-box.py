def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    pass

    def area(box) -> int:
        if box[2] <= box[0] or box[3] <= box[1]:
            return 0
        return (box[2] - box[0]) * (box[3] - box[1])

    in_x1, in_x2 = max(box_a[0], box_b[0]), min(box_a[2], box_b[2])
    in_y1, in_y2 = max(box_a[1], box_b[1]), min(box_a[3], box_b[3])

    box_in = [in_x1, in_y1, in_x2, in_y2]
    s_in = area(box_in)
    s_un = area(box_a) + area(box_b) - s_in
    return s_in / s_un