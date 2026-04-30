import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    mode = np.argmax(np.bincount(np.asarray(y_train)))
    n = len(X_test)
    return np.repeat(mode, n)