def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    n = len(values)
    cnt = {}
    for v in values:
        if v not in cnt.keys():
            cnt.update({v : 1})
        else:
            cnt[v] += 1

    return [cnt[v] / n for v in values]