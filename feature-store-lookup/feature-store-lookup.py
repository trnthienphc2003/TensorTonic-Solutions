def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    # Write code here
    ans = []
    for req in requests:
        if req["user_id"] not in feature_store.keys():
            feature_store.update({
                req["user_id"] : defaults
            })

        for online_feat in req["online_features"].keys():
            feature_store[req["user_id"]].update({
                online_feat: req["online_features"][online_feat]
            })

        ans.append(feature_store[req["user_id"]])
    return ans