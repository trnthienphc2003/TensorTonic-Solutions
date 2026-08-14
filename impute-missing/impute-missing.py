import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    If an entire column is NaN, fill it with 0.
    """
    X_imputed = np.array(X, dtype=np.float64, copy=True)
    
    # 1. Compute summary statistic per column ignoring NaNs
    stat_func = np.nanmean if strategy == 'mean' else np.nanmedian
    
    with np.errstate(all='ignore'):
        col_stats = stat_func(X_imputed, axis=0)
    
    # 2. Replace NaN statistics (from all-NaN columns) with 0.0
    col_stats = np.nan_to_num(col_stats, nan=0.0)
    
    # 3. Use 2D broadcasting via np.where to fill NaNs directly
    nan_mask = np.isnan(X_imputed)
    X_imputed = np.where(nan_mask, col_stats, X_imputed)
    
    return X_imputed