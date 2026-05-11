def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    # Write code here
    st = max(lags)
    lag_feat_matrix = []
    for i in range(st, len(series)):
        lag_feat_matrix.append([series[i - t] for t in lags])
    return lag_feat_matrix