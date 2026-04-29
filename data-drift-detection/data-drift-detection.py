def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    n = len(reference_counts)
    tot_ref, tot_prod = sum(reference_counts), sum(production_counts)
    tvd = .5 * sum([abs(reference_counts[i] / tot_ref - production_counts[i] / tot_prod) for i in range(n)])
    return {
        'score': tvd,
        'drift_detected': tvd > threshold
    }