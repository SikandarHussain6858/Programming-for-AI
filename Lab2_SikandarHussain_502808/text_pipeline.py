import numpy as np


def build_vocabulary(texts):
    vocab = {}
    idx = 0
    for text in texts:
        words = text.lower().split()
        for word in words:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def text_to_ids(text, vocab):
    words = text.lower().split()
    return [vocab[word] for word in words if word in vocab]


def ids_to_bow(ids, vocab_size):
    vec = np.zeros(vocab_size)
    for idx in ids:
        vec[idx] += 1
    return vec


def batch_text_to_bow(texts, vocab):
    vocab_size = len(vocab)
    bow_vectors = []
    for text in texts:
        ids = text_to_ids(text, vocab)
        bow = ids_to_bow(ids, vocab_size)
        bow_vectors.append(bow)
    return np.vstack(bow_vectors)


def linear_text_scoring(bow_matrix, weights):
    return bow_matrix @ weights
