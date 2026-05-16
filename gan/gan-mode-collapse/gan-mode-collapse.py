import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    Returns: dict with "diversity_score" (float) and "is_collapsed" (bool)
    """
    # Your implementation here
    generated_samples = np.asarray(generated_samples)
    sigma = np.std(generated_samples, axis=0)
    div_score = np.average(sigma)

    return {
        'diversity_score': div_score,
        'is_collapsed': div_score < threshold
    }