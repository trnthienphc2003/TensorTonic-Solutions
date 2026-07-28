def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    # Write code here
    R = [list(filter(lambda x: x != 0, row)) for row in matrix]
    mean_R = [sum(row) / len(row) if len(row) != 0 else 0.0 for row in R]
    matrix_norm = [[matrix[i][j] - mean_R[i] if matrix[i][j] != 0 else 0 for j in range(len(matrix[i]))] for i in range(len(matrix))]
    return matrix_norm