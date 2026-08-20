import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    N = len(documents)
    vocab = {}
    documents = [doc.split() for doc in documents]

    for doc in documents:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    V = len(vocab)
    tf = np.zeros((V, N), dtype=np.float64)
    df = np.zeros((V,), dtype=np.float64)

    for d, doc in enumerate(documents):
        counts = Counter(doc)
        for word, cnt in counts.items():
            v = vocab[word]
            tf[v, d] += cnt / len(doc)
            df[v] += 1

    idf = np.log(N / df)
    tf_idf = tf * idf[..., None]

    idx = np.argsort(list(vocab.keys()))
    # assert False, tf_idf.shape
    return tf_idf[idx].T, sorted(list(vocab.keys()))