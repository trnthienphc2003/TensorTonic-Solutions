import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    var_res = np.sum((y_true - y_pred) ** 2, axis=-1)
    var_tot = np.sum((y_true - y_true.mean()) ** 2, axis = -1)
    if var_tot == 0.:
        if var_res == 0.:
            return 1.
        else:
            return 0.

    return 1. - var_res / var_tot