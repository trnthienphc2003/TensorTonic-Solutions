import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    # Write code here
    N_class = max(y_train) + 1
    prob = [0.0 for _ in range(N_class)]
    N, D = len(y_train), len(X_train[0])
    
    for c in y_train:
        prob[c] += 1
    prob = list(map(lambda x: x / N, prob))

    mean = [[0.0 for _ in range(D)] for _ in range(N_class)]
    var = [[0.0 for _ in range(D)] for _ in range(N_class)]
    for i in range(N):
        c = y_train[i]
        for j in range(D):
            mean[c][j] += X_train[i][j]

    for c in range(N_class):
        for d in range(D):
            mean[c][d] /= (prob[c] * N)

    for i in range(N):
        c = y_train[i]
        for j in range(D):
            var[c][j] += (X_train[i][j] - mean[c][j]) ** 2

    for c in range(N_class):
        for d in range(D):
            var[c][d] /= (prob[c] * N)
            
    log_P = list(map(math.log, prob))

    N_test = len(X_test)
    eps = 1e-8
    ans = []
    for i in range(N_test):
        log_pos = 0.0
        best_log, best_class = float("-inf"), 0
        for c in range(N_class):
            log_pos = log_P[c]
            for j in range(D):
                v = var[c][j] + eps
                log_pos -= .5 * math.log(2. * math.pi * v)
                log_pos -= ((X_test[i][j] - mean[c][j]) ** 2) / (2. * v)

            if(log_pos > best_log):
                best_log = log_pos
                best_class = c
        # if i == 1:
        #     assert False, f'{best_log} {best_class}'
        ans.append(best_class)

    return ans