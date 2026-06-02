import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    if len(vocab) == 0:
        return np.asarray([], dtype=int)
    return np.array([tokens.count(word) for word in vocab], dtype=int)