import numpy as np
cimport numpy as np
import json
import random
from collections import defaultdict, OrderedDict
from types import SimpleNamespace
from tqdm.notebook import tqdm
import time


# === corpus loading ===
cdef class NeurIPSCorpus:
    cdef np.ndarray docs
    cdef np.ndarray doc_lengths
    cdef dict word2id
    cdef int num_docs
    cdef int num_topics
    cdef int num_words
    cdef int max_doc_length

    def __init__(self, state_dict_path):
        cdef dict state_dict = json.load(open(state_dict_path))
        cdef list docs = state_dict["docs"]
        self.word2id = state_dict["word2id"]
        self.num_topics = state_dict["num_topics"]
        self.max_doc_length = state_dict["max_doc_length"]
        self.num_docs = len(docs)
        self.num_words = len(self.word2id)
        self.docs = np.zeros([self.num_docs, self.max_doc_length], dtype=int)
        self.doc_lengths = np.zeros([self.num_docs], dtype=int)
        cdef int i, j
        for i in range(self.num_docs):
            for j in range(len(docs[i])):
                self.docs[i, j] = docs[i][j]
            self.doc_lengths[i] = len(docs[i])
        print(
            "num_docs:", self.num_docs, 
            "num_topics:", self.num_topics, 
            "num_words:", self.num_words
        )

cdef class Hyperparameters:
    cdef np.ndarray alpha
    cdef np.ndarray beta
    cdef int gibbs_sampling_max_iters

    def __init__(self, alpha, beta, gibbs_sampling_max_iters):
        self.alpha = alpha
        self.beta = beta 
        self.gibbs_sampling_max_iters = gibbs_sampling_max_iters

cdef NeurIPSCorpus corpus = NeurIPSCorpus("data/papers.json")
cdef Hyperparameters hparams = Hyperparameters(
    alpha=np.ones([corpus.num_topics], dtype=float) / corpus.num_topics,
    beta=np.ones([corpus.num_words], dtype=float) / corpus.num_topics,
    gibbs_sampling_max_iters=10,
)

start_time = time.time()

# === initialization ===
print("Initializing...")
cdef np.ndarray n_doc_topic = np.zeros([corpus.num_docs, corpus.num_topics], dtype=float) # n_m^(k)
cdef np.ndarray n_topic_word = np.zeros([corpus.num_topics, corpus.num_words], dtype=float) # n_k^(t)
cdef np.ndarray z_doc_word = np.zeros([corpus.num_docs, corpus.max_doc_length], dtype=int)

cdef int topic_ij, new_topic_ij
cdef int doc_i, j, word_j, iteration
cdef np.ndarray p_topic = np.zeros([corpus.num_topics], dtype=float)
cdef np.ndarray p_doc_topic = np.zeros([corpus.num_topics], dtype=float)
cdef np.ndarray p_topic_word = np.zeros([corpus.num_topics], dtype=float)
cdef np.ndarray phi = np.zeros([corpus.num_docs, corpus.num_topics], dtype=float)
cdef np.ndarray theta = np.zeros([corpus.num_topics, corpus.num_words], dtype=float)

topics = np.random.randint(0, corpus.num_topics, 
    size=(corpus.num_docs, corpus.max_doc_length))
for doc_i in range(corpus.num_docs):
    topics_i = np.random.randint(0, corpus.num_topics, 
    for j in range(corpus.doc_lengths[doc_i]):
        word_j = corpus.docs[doc_i, j]
        topic_ij = topics[doc_i, j]
        n_doc_topic[doc_i, topic_ij] += 1
        n_topic_word[topic_ij, word_j] += 1
        z_doc_word[doc_i, j] = topic_ij

print(time.time() - start_time)

# === Gibbs sampling ===
print("Gibbs sampling...")
for iteration in range(hparams.gibbs_sampling_max_iters):
    print(f"Iter [{iteration}] ===")

    for doc_i in range(corpus.num_docs):
        for j, word_j in enumerate(corpus.docs[doc_i]):
            # remove the old assignment
            topic_ij = z_doc_word[doc_i, j]
            n_doc_topic[doc_i, topic_ij] -= 1
            n_topic_word[topic_ij, word_j] -= 1
            # compute the new assignment
            p_doc_topic = (n_doc_topic[doc_i, :] + hparams.alpha) \
                        / np.sum(n_doc_topic[doc_i] + hparams.alpha)
            p_topic_word = (n_topic_word[:, word_j] + hparams.beta[word_j]) \
                        / np.sum(n_topic_word + hparams.beta, axis=1)
            p_topic = p_doc_topic * p_topic_word
            p_topic /= np.sum(p_topic)
            # record the new assignment
            new_topic_ij = np.random.choice(np.arange(corpus.num_topics), p=p_topic)
            n_doc_topic[doc_i, new_topic_ij] += 1
            n_topic_word[new_topic_ij, word_j] += 1
            z_doc_word[doc_i, j] = new_topic_ij

    # === Check convergence and read out parameters ===
    theta = (n_doc_topic + hparams.alpha) / np.sum(n_doc_topic + hparams.alpha, axis=1, keepdims=True)
    phi = (n_topic_word + hparams.beta) / np.sum(n_topic_word + hparams.beta, axis=1, keepdims=True)
    print("theta:\n", theta, "\nphi:\n", phi)