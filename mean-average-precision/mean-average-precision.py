import numpy as np

def mean_average_precision(y_true_list: list, y_score_list: list, k: int | None = None) -> dict:
    """
    Returns a dictionary with map_value and ap_per_query.
    """
    # Write code here
    mx_len = max(len(y) for y in y_true_list)

    y_true_list = np.array([
        np.pad(lst, (0, mx_len - len(lst)), constant_values=0)
        for lst in y_true_list
    ])

    y_score_list = np.array([
        np.pad(lst, (0, mx_len - len(lst)), constant_values=0.0)
        for lst in y_score_list
    ])
    # y_true_list = np.asarray(y_true_list)
    # y_score_list = np.asarray(y_score_list)

    Q, N = y_true_list.shape
    
    rank = np.argsort(y_score_list, axis=-1)[..., ::-1]
    if k is not None:
        rank = rank[..., :k]
        N = k
    # assert False, rank
    
    R = np.sum(y_true_list, axis=-1)
    x = np.repeat(np.arange(Q), N)
    y = rank.flatten()
    sorted_true = y_true_list[x, y].reshape((Q, N))
    # assert False, sorted_true
    # sorted_rel = 

    pos = np.arange(1, N + 1)
    prec = np.cumsum(sorted_true, axis=-1) * sorted_true / pos[None]

    prec = np.nan_to_num(prec, 0.0)
    AP = prec.sum(axis=-1) / R
    AP = np.nan_to_num(AP, 0.0)
    # assert False, AP
    mAP = AP.mean() if AP.any() else 0.0
    return {
        'map_value': mAP,
        'ap_per_query': AP.tolist()
    }