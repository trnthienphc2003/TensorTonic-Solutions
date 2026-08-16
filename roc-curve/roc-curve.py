import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    
    thresholds = np.append(np.unique(y_score), [float("inf")])
    thresholds = np.sort(thresholds)[::-1]
    # assert False, thresholds

    tpr, fpr = [], []
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        conf_mat = np.zeros((2, 2))
        np.add.at(
            conf_mat,
            (y_true, y_pred),
            1
        )

        tp = conf_mat[1, 1]
        fp = conf_mat[0, 1]
        fn = conf_mat[1, 0]
        tn = conf_mat[0, 0]

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
        # assert False, conf_mat

    return fpr, tpr, thresholds