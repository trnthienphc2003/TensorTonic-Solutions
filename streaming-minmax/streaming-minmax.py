import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    # Write code here
    return {
        'min': np.full((D,), float("inf")),
        'max': np.full((D,), float("-inf"))
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    # Write code here
    X_batch = np.asarray(X_batch, dtype=np.float64)
    mn = np.minimum(state['min'], X_batch.min(axis=0))
    mx = np.maximum(state['max'], X_batch.max(axis=0))

    X_norm = (X_batch - mn) / (mx - mn + eps)
    state['min'], state['max'] = mn, mx

    return X_norm