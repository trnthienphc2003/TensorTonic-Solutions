from collections import Counter
import numpy as np


def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    counts = Counter(tokens)
    return np.asarray([counts[word] for word in vocab], dtype=int)