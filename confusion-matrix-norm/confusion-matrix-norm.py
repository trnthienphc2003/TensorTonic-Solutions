import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    conf_mat = np.zeros((num_classes, num_classes))
    np.add.at(conf_mat, (y_true, y_pred), 1)
    if normalize == 'true':
        conf_mat /= np.sum(conf_mat, axis=-1, keepdims=True)
    elif normalize == 'pred':
        conf_mat /= np.sum(conf_mat, axis=0, keepdims=True)
    elif normalize == 'all':
        conf_mat /= np.sum(conf_mat)
    else:
        assert normalize == 'none', f"The normalize strategy '{normalize}' does not exist"
        pass

    return conf_mat