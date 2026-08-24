def baseline_predict(ratings_matrix: list, target_pairs: list) -> list:
    """
    Returns the baseline predictions for the requested user-item pairs.
    """
    # Write code here
    n_users, n_items = len(ratings_matrix), len(ratings_matrix[0])
    user_mean = [0 for _ in range(n_users)]
    item_mean = [0 for _ in range(n_items)]
    valid = 0

    for idx, user_rating in enumerate(ratings_matrix):
        valid = 0
        for rated_score in user_rating:
            if rated_score != 0:
                user_mean[idx] += rated_score
                valid += 1
        user_mean[idx] = (user_mean[idx] / valid) if valid > 0 else 0.0

    for idx in range(n_items):
        valid = 0
        for u_idx in range(n_users):
            if ratings_matrix[u_idx][idx] != 0:
                item_mean[idx] += ratings_matrix[u_idx][idx]
                valid += 1
        item_mean[idx] = (item_mean[idx] / valid) if valid > 0 else 0.0
    
    global_mu = sum(user_mean) / n_users
    bias_u = [user_mean[i] - global_mu for i in range(n_users)]
    bias_i = [item_mean[i] - global_mu for i in range(n_items)]

    target_user, target_item = zip(*target_pairs)
    target_user, target_item = list(target_user), list(target_item)
    n_queries = len(target_pairs)

    predict_score = [0 for _ in range(n_queries)]
    for q_idx in range(n_queries):
        predict_score[q_idx] = global_mu + bias_u[target_user[q_idx]] + bias_i[target_item[q_idx]]

    return predict_score

