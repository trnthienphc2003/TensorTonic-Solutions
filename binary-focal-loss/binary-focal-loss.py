from math import log

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here

    n_sample = len(predictions)
    assert n_sample == len(targets)
    pred = [(predictions[i] if targets[i] == 1 else 1. - predictions[i]) for i in range(n_sample)]
    # pred = [predictions[i] for i in range(n_sample) if targets[i] == 1 else 0]

    logits = [alpha * log(pred[i]) * ((1. - pred[i]) ** gamma) for i in range(n_sample)]
    return -sum(logits) / n_sample