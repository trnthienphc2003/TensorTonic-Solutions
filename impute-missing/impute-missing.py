import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X_norm = np.asarray(X, dtype=np.float64)

    # reduce_mode = (np.mean if strategy == 'mean' else np.median)
    X_cannot_impute = np.all(np.isnan(X), axis=0)
    X_can_impute = np.logical_not(X_cannot_impute)
    X_valid = np.logical_not(np.isnan(X))
    # assert False, X_valid

    fill_value = None
    if strategy == 'mean':
        fill_value = np.mean(
            X_norm[..., X_can_impute],
            axis=0,
            where=np.logical_not(np.isnan(X_norm[..., X_can_impute]))
        )
    else:
        fill_value = np.median(
            X_norm[X_valid],
            axis=0
        )
    X_norm[..., X_can_impute] = np.where(
        np.isnan(X_norm[..., X_can_impute]),
        fill_value,
        X_norm[..., X_can_impute]
    )

    X_norm[..., X_cannot_impute] = 0
    return X_norm