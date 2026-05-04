import itertools

def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    # Write code here
    return len(list(set(itertools.chain.from_iterable(recommendations)))) / n_items