import numpy as np
import math

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    # Write code here
    results = {}
    feat_keys = train_dist.keys()
    for feat in feat_keys:
        train = train_dist.get(feat)
        serve = serving_dist.get(feat)

        psi = 0.
        for i in range(len(train)):
            t = train[i] + eps
            s = serve[i] + eps
            psi += (s - t) * (math.log(s / t))
        results[feat] = {
            'psi': psi,
            'skewed': psi >= threshold
        }

    return results