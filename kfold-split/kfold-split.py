import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns:
        list of k tuples: (train_idx, val_idx)
    """
    idx = np.arange(N)

    if shuffle:
        if rng is None:
            np.random.shuffle(idx)
        else:
            rng.shuffle(idx)

    folds = np.array_split(idx, k)

    splits = []
    for i in range(k):
        val_idx = folds[i]

        train_idx = np.concatenate(
            folds[:i] + folds[i+1:]
        )

        splits.append((train_idx, val_idx))

    return splits