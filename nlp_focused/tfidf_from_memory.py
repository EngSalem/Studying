
from collections import defaultdict, Counter
import numpy as np

class TfIDf:
    def __init__(self, corpus, vocab=None):
        self.corpus = corpus
        self.N = len(corpus)
        if vocab is None:
            self.vocab = self.build_vocab(corpus)
        else:
            self.build_vocab(corpus)
        
        self.get_df()

        def build_vocab(self, corpus):
            self.vocab = set()
            for doc in corpus:
                tokens = self.tokenize(doc)
                self.vocab.update(tokens)
            self.vocab = sorted(self.vocab)

        def tokenize(self, text):
            return text.lower().split()
        

        def get_df(self):
            self.df = defaultdict(int)
            for doc in self.corpus:
                # get unique tokens in doc
                tokens  = set(self.tokenize(doc))
                for token in tokens:
                    self.df[token] +=1

        def get_idf(self, token):
            self.token_idf = np.log((1 + self.N) / (1 + self.df.get(token, 0))) + 1

        def fit(self):
            # obtain get document frequencies and idf for each token
            self.tf_idf = np.zeros((self.N, len(self.vocab)))
            for i, doc in enumerate(self.corpus):
                tokens = self.tokenize(doc)
                tf = Counter(tokens)
                for j, token in enumerate(self.vocab):
                    self.tf_idf[i,j] = tf.get(token, 0) * self.get_idf(token)

        def transform(self, new_corpus):
            new_tf_idf = np.zeros((len(new_corpus), len(self.vocab)))
            for i, doc in enumerate(new_corpus):
                tokens = self.tokenize(doc)
                tf = Counter(tokens)
                for j, token in enumerate(self.vocab):
                    new_tf_idf[i,j] = tf.get(token, 0) * self.get_idf(token)
            return new_tf_idf            



            

            