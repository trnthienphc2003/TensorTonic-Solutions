def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    metrics = {}
    N = len(y_true)
    assert N == len(y_pred)
    if system_type == 'classification':
        metrics = {
            'accuracy': 0.,
            'precision': 0.,
            'recall': 0.,
            'f1': 0.,
        }

        tp = sum([1 if y_true[i] == y_pred[i] and y_true[i] == 1 else 0 for i in range(N)])
        tn = sum([1 if y_true[i] == y_pred[i] and y_true[i] == 0 else 0 for i in range(N)])
        fp = sum([1 if y_true[i] != y_pred[i] and y_true[i] == 1 else 0 for i in range(N)])
        fn = sum([1 if y_true[i] != y_pred[i] and y_true[i] == 0 else 0 for i in range(N)])

        metrics['accuracy'] = (tp + tn) / N
        metrics['precision'] = (tp / (tp + fp) if tp + fp != 0 else 0.)
        metrics['recall'] = (tp / (tp + fn) if tp + fn != 0 else 0.)
        metrics['f1'] = (2. * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if metrics['precision'] + metrics['recall'] != 0. else 0.)
    elif system_type == 'regression':
        metrics = {
            'mae': 0.,
            'rmse': 0.,
        }

        metrics['mae'] = sum([abs(y_pred[i] - y_true[i]) for i in range(N)]) / N
        metrics['rmse'] = (sum([(y_pred[i] - y_true[i]) ** 2 for i in range(N)]) / N) **.5
    elif system_type == 'ranking':
        metrics = {
            'precision_at_3': 0.,
            'recall_at_3': 0.,
        }


        rel_idx = argsort(y_pred)
        rel_idx = rel_idx[::-1][:3]
        rel_cnt = sum(y_true)
        rel_top_3 = sum([y_true[x] for x in rel_idx])

        metrics['precision_at_3'] = rel_top_3 / 3
        metrics['recall_at_3'] = rel_top_3 / rel_cnt if rel_cnt != 0. else 0.
    else:
        
        raise ValueError(f'System type {system_type} is not defined')
        return None

    return sorted(metrics.items())