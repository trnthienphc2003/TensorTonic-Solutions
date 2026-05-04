from math import log

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    K = len(predictions)
    q = [(1. - epsilon) + epsilon / K if i == target else epsilon / K for i in range(K)]
    return sum([-q[i] * log(predictions[i]) for i in range(K)])