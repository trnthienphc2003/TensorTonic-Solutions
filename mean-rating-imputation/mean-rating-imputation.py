import numpy as np

def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    # Write code here
    n_user, n_item = len(ratings_matrix), len(ratings_matrix[0])
    ratings_matrix = np.asarray(ratings_matrix, dtype=np.float64)
    mask = (ratings_matrix == 0)
    ratings_matrix[mask] = np.nan
    axis = -1 if mode == 'user' else 0

    impute_value = np.nanmean(ratings_matrix, axis=axis, keepdims=True)
    impute_value = np.nan_to_num(impute_value, nan=0.0)

    ratings_matrix = np.where(mask, impute_value, ratings_matrix)
    return ratings_matrix.tolist()