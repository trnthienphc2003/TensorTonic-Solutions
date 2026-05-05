def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    cnt = {}
    sum = {}
    for idx, c in enumerate(categories):
        if c not in cnt.keys():
            cnt.update({
                c : 1
            })
            sum.update({
                c : targets[idx]
            })
        else:
            cnt[c] += 1
            sum[c] += targets[idx]
    return [sum[c] / cnt[c] for c in categories]