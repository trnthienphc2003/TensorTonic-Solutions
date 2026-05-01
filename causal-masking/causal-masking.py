def apply_causal_mask(scores, mask_value=-1e9):
    scores = np.asarray(scores, dtype=np.float64)

    T = scores.shape[-1]
    
    # Create causal mask: True where we KEEP values
    mask = np.tril(np.ones((T, T), dtype=bool))

    # Apply mask via broadcasting
    return np.where(mask, scores, mask_value)