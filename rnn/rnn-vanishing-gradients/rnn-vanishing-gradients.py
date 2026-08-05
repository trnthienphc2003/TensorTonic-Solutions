import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate maximum gradient norm decay/explosion factor over T time steps.
    
    The spectral norm of W_hh^k is upper-bounded by ||W_hh||_2^k = (sigma_max)^k.
    Returns list of upper-bound gradient scale factors for step 0 to T-1 back in time.
    """
    # 1. Compute the spectral norm (largest singular value)
    sigma_max = np.linalg.norm(W_hh, ord=2)  # Equivalent to np.linalg.svd(W_hh, compute_uv=False)[0]
    
    # 2. Compute powers (sigma_max^0, sigma_max^1, ..., sigma_max^(T-1)) safely
    steps = np.arange(T, dtype=np.float64)
    
    if sigma_max == 0:
        norms = np.zeros(T, dtype=np.float64)
        if T > 0:
            norms[0] = 1.0
        return norms.tolist()
        
    # Log-space power to prevent overflow issues before converting back
    log_norms = steps * np.log(sigma_max)
    norms = np.exp(log_norms)
    
    return norms.tolist()