def frequency_encoding(values):
    n = len(values)
    cnt = {}
    
    for v in values:
        cnt[v] = cnt.get(v, 0) + 1

    return [cnt[v] / n for v in values]