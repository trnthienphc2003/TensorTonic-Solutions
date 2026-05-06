from datetime import datetime

def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    # Write code here
    # assert False, -datetime.strptime(models[0]['timestamp'], '%Y-%m-%d').timestamp()
    models.sort(key = lambda x: (-x['accuracy'], x['latency'], -datetime.strptime(x['timestamp'], '%Y-%m-%d').timestamp()))
    return models[0]['name']