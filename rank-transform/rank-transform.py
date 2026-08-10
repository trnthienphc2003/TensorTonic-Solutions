from collections import defaultdict

def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    cnt, pos = defaultdict(int), defaultdict(int)
    actual_rank = 1
    prev_item = 0

    sort_val = sorted(values)
    for v in sort_val:
        pos[v] += actual_rank
        cnt[v] += 1
        actual_rank += 1
        prev_item = v

    return ([pos[v] / cnt[v] for v in values])