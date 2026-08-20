import numpy as np

def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    # Write code here
    n_user, n_item = len(ratings_matrix), len(ratings_matrix[0])
    ratings_matrix = np.asarray(ratings_matrix)
    # fill_mask = np.where(ratings_matrix == 0)
    # assert False, ratings_matrix
    
    if mode == 'user':
        valid_cnt = np.count_nonzero(ratings_matrix, axis=-1, keepdims=True)
        impute_value = np.sum(ratings_matrix, axis=-1, keepdims=True) / valid_cnt
        impute_value = np.repeat(impute_value, repeats=n_item, axis=-1)
        # assert False, impute_value
        ratings_matrix = np.where(
            ratings_matrix == 0,
            impute_value,
            ratings_matrix
        )

    else:
        valid_cnt = np.count_nonzero(ratings_matrix, axis=0, keepdims=True)
        impute_value = np.sum(ratings_matrix, axis=0, keepdims=True) / valid_cnt
        impute_value = np.repeat(impute_value, repeats=n_user, axis=0)
        # assert False, impute_value
        ratings_matrix = np.where(
            ratings_matrix == 0,
            impute_value,
            ratings_matrix
        )

    # assert False, ratings_matrix
    ratings_matrix = np.nan_to_num(ratings_matrix, nan=0.)
    return ratings_matrix.tolist()