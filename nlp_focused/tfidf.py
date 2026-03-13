# ---------------------------------------------------------------
# TIER 1 — IMPLEMENT FROM SCRATCH
# ---------------------------------------------------------------

# PROBLEM 1: TF-IDF Vectorizer
# Difficulty: Medium
# ---------------------------------------------------------------
# You are given a list of Zillow listing descriptions.
# Implement TF-IDF vectorization from scratch.
#
# Requirements:
#   - Implement tf(term, doc): term frequency of a term in a document
#   - Implement idf(term, corpus): log((1 + N) / (1 + df(term))) + 1  [sklearn-style, smoothed]
#   - Implement tfidf_matrix(corpus): returns an (N x V) matrix where
#     N = number of docs, V = vocabulary size
#   - Tokenize by whitespace, lowercase, ignore punctuation
#
# Example:
listings = [
    "spacious 3 bedroom house with large backyard",
    "cozy 1 bedroom apartment near downtown",
    "3 bedroom house with garage and pool",
]
# Expected: a 3 x vocab_size numpy matrix of tfidf scores
import numpy as np

class tdfidf():
    def __init__(self, corpus, vocab=None):
        self.corpus = corpus
        self.vocab = vocab if vocab is not None else self.build_vocab(corpus)
        self.N = len(corpus) # number of documents
        self.df = self.compute_df(corpus)
    
    def build_vocab(self, corpus):
        vocab = set()
        for doc in corpus:
            tokens = self.tokenize(doc)
            vocab.update(tokens)
        return sorted(vocab)
    
    def tokenize(self, text):
        text = text.lower()
        tokens = text.split()
        return tokens

    def tf(self, term, doc):
        tokens = self.tokenize(doc)
        return tokens.count(term)    
    
    def idf(self, term):
        df_term = self.df.get(term, 0)
        return np.log((1+ self.N)) / (1 + df_term) + 1 # 1 is for smoothing
    
    def compute_df(self, corpus):
        df  = {}
        for doc in corpus:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        return df
    
    def tfidf_matrix(self):
        V = len(self.vocab)
        tfidf_mat = np.zeros((self.N, V))
        for i, doc in enumerate(self.corpus):
            for j, term in enumerate(self.vocab):
                tfidf_mat[i, j] = self.tf(term, doc) * self.idf(term)
        return tfidf_mat
    
    def fit_transform(self):
        return self.tfidf_matrix()
    
    def transform(self, new_corpus):
        new_tfidf_mat = np.zeros((len(new_corpus), len(self.vocab)))
        for i, doc in enumerate(new_corpus):
            for j, term in enumerate(self.vocab):
                new_tfidf_mat[i, j] = self.tf(term, doc) * self.idf(term)
        return new_tfidf_mat
 
    