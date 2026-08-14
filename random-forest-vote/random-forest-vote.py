import numpy as np
from scipy import stats

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    m, cnt = stats.mode(predictions, axis=0)
    return m.tolist()