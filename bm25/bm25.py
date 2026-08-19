import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    # Write code here
    N = len(docs)
    vocab = {}
    for doc in docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    q = np.array(
        [vocab[token] for token in query_tokens if token in vocab]
    )

    if len(q) == 0:
        return np.zeros(N)

    V = len(vocab)
    tf = np.zeros((V, N), dtype=np.float64)
    df = np.zeros((V,), dtype=np.float64)

    for d, doc in enumerate(docs):
        counts = Counter(doc)
        for word, count in counts.items():
            v = vocab[word]
            tf[v, d] += count
            df[v] += 1

    idf = np.log((N - df + .5) / (df + .5) + 1)
    doc_len = np.array([len(doc) for doc in docs], dtype=np.float64)
    norm = k1 * (1 - b + b * doc_len[:, None] / doc_len.mean())

    f = tf[q].T

    score = idf[q][None, ...] * (
        f * (k1 + 1) / (f + norm)
    )

    return score.sum(axis=-1)