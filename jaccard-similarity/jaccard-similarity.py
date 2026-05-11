def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    if len(set_a) == 0 and len(set_b) == 0:
        return 0

    set_a = list(set(set_a))
    set_b = list(set(set_b))
    set_in = list(set(set_a).intersection(set(set_b)))
    # assert False, set_in
    return len(set_in) / (len(set_a) + len(set_b) - len(set_in))