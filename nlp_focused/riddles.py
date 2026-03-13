# ---------------------------------------------------------------
# TIER 1 — IMPLEMENT FROM SCRATCH
# ---------------------------------------------------------------

# PROBLEM 1: TF-IDF Vectorizer
# Difficulty: Medium
# ---------------------------------------------------------------

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


# PROBLEM 2: Cosine Similarity Search
# Difficulty: Easy-Medium
# ---------------------------------------------------------------
# Given a query and a corpus of listing descriptions,
# return the top-k most similar listings using cosine similarity
# on TF-IDF vectors (you can reuse Problem 1).
#
# def cosine_similarity(vec_a, vec_b) -> float
# def search(query, corpus, k=3) -> list of (listing, score)
#
# Example:
query = "3 bedroom house garage"
# Expected: listings ranked by cosine similarity to query


# PROBLEM 3: BM25 Ranking
# Difficulty: Hard
# ---------------------------------------------------------------
# Zillow's search engine ranks listings using BM25.
# Implement BM25 scoring from scratch.
#
# BM25 formula per query term qi:
#   score(D, Q) = Σ IDF(qi) * f(qi,D)*(k1+1) / (f(qi,D) + k1*(1 - b + b*|D|/avgdl))
#
#   IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
#   k1 = 1.5,  b = 0.75
#   f(qi, D) = term freq of qi in D
#   |D|      = number of words in D
#   avgdl    = average document length across corpus
#
# def bm25_rank(corpus, query) -> list of (listing, score) sorted desc
#
# Example:
corpus = [
    "3 bedroom house with garage and large backyard",
    "cozy 1 bedroom apartment near downtown",
    "spacious 3 bedroom home with garage and pool",
]
query = "3 bedroom garage"
# Expected: listing 0 and listing 2 rank above listing 1


# PROBLEM 4: Byte Pair Encoding (BPE) — Core Loop
# Difficulty: Hard
# ---------------------------------------------------------------
# Implement the core BPE training loop from scratch.
#
# Steps:
#   1. Start with character-level tokenization of the vocabulary
#      e.g. "low" → ["l", "o", "w", "</w>"]
#   2. Count all adjacent symbol pairs across the corpus
#   3. Merge the most frequent pair into a new symbol
#   4. Repeat for `num_merges` iterations
#   5. Return the list of merge rules in order
#
# def train_bpe(corpus: list[str], num_merges: int) -> list[tuple]
#
# Example:
corpus = ["low low low lower lowest", "newer newer wider"]
# After a few merges you might see: ('l','o') → 'lo', ('lo','w') → 'low', etc.


# PROBLEM 5: Edit Distance (Levenshtein)
# Difficulty: Medium
# ---------------------------------------------------------------
# Zillow deals with messy address inputs from users.
# Implement edit distance to measure how different two address
# strings are, using dynamic programming.
#
# Operations allowed: insert, delete, substitute (each costs 1)
#
# def edit_distance(s1: str, s2: str) -> int
#
# Bonus: implement a function that given a raw address string
# and a list of canonical addresses, returns the closest match.
#
# Example:
s1 = "123 Mian St"   # user typo
s2 = "123 Main St"   # canonical
# Expected: edit_distance(s1, s2) == 1


# ---------------------------------------------------------------
# TIER 2 — APPLIED NLP (REAL-WORLD ZILLOW PROBLEMS)
# ---------------------------------------------------------------

# PROBLEM 6: NER — Extract Listing Features
# Difficulty: Medium
# ---------------------------------------------------------------
# Given a raw listing description, extract structured features
# using regex and rule-based NER (no ML models).
#
# Extract:
#   - bedrooms (int)
#   - bathrooms (float)
#   - price (int)
#   - amenities (list of strings from a fixed set)
#
# AMENITY_SET = {"garage", "pool", "backyard", "gym", "fireplace",
#                "balcony", "basement", "parking"}
#
# def extract_features(description: str) -> dict
#
# Example:
desc = "Beautiful 3 bedroom, 2.5 bathroom home listed at $540,000. " \
       "Features include a garage, pool, and large backyard."
# Expected:
# {
#   "bedrooms": 3,
#   "bathrooms": 2.5,
#   "price": 540000,
#   "amenities": ["garage", "pool", "backyard"]
# }


# PROBLEM 7: Duplicate Listing Detection (SimHash)
# Difficulty: Hard
# ---------------------------------------------------------------
# Zillow receives many near-duplicate listing descriptions from
# different agents for the same property.
# Implement SimHash to detect near-duplicates efficiently.
#
# Steps:
#   1. Tokenize and hash each token using a hash function
#   2. For each bit position (use 64 bits), sum +1 if bit is set, -1 if not,
#      weighted by token frequency
#   3. Final simhash: bit = 1 if sum > 0 else 0
#   4. Hamming distance between two simhashes < threshold → near-duplicate
#
# def simhash(text: str) -> int         # returns 64-bit integer
# def hamming(h1: int, h2: int) -> int  # number of differing bits
# def are_duplicates(t1, t2, threshold=3) -> bool
#
# Example:
t1 = "Stunning 3 bedroom house with garage and large pool in Seattle"
t2 = "Stunning 3 bedroom house with garage and large pool in Seattle WA"
t3 = "1 bedroom apartment near downtown Portland, great city views"
# Expected: are_duplicates(t1, t2) == True, are_duplicates(t1, t3) == False


# PROBLEM 8: Listing Description Summarizer (Extractive)
# Difficulty: Medium
# ---------------------------------------------------------------
# Given a long listing description, extract the top-k most
# informative sentences using TF-IDF sentence scoring.
#
# Steps:
#   1. Split description into sentences
#   2. Compute a TF-IDF representation for each sentence
#      using the full description as the corpus
#   3. Score each sentence by the sum of its TF-IDF values
#   4. Return the top-k sentences in their original order
#
# def extractive_summarize(description: str, k: int = 3) -> str
#
# Example:
long_desc = """
This stunning property is located in the heart of Seattle.
It features 4 bedrooms and 3 modern bathrooms.
The kitchen was recently renovated with quartz countertops.
There is a two-car garage and a spacious backyard with a deck.
The home is walking distance to top-rated schools and parks.
Monthly HOA fees are $250 and include landscaping.
"""
# Expected: top 3 most informative sentences joined as a string


# PROBLEM 9: Semantic Chunking for RAG
# Difficulty: Medium
# ---------------------------------------------------------------
# You are building a Q&A system over long property documents
# (inspection reports, HOA agreements). Before passing to an LLM,
# you must chunk the text intelligently.
#
# Implement TWO chunking strategies and compare them:
#
# Strategy A — Fixed-size chunking:
#   Split text into chunks of max `chunk_size` words,
#   with `overlap` words between consecutive chunks.
#
# Strategy B — Sentence-boundary chunking:
#   Group sentences into chunks such that each chunk stays
#   within max `max_words` words. Do not split mid-sentence.
#
# def fixed_chunk(text, chunk_size=100, overlap=20) -> list[str]
# def sentence_chunk(text, max_words=100) -> list[str]
#
# Bonus: given a query, retrieve the most relevant chunk
# using cosine similarity on TF-IDF vectors.


# PROBLEM 10: Intent Classifier — Zero-Shot with Embeddings
# Difficulty: Medium-Hard
# ---------------------------------------------------------------
# Zillow's search assistant needs to classify user queries
# into intents WITHOUT any labeled training data.
#
# Intents: ["buy", "rent", "estimate_value", "find_agent", "general_info"]
#
# Approach:
#   1. Represent each intent as a short natural language description
#      e.g. "buy" → "I want to purchase a home"
#   2. Represent the query the same way
#   3. Compute cosine similarity between query and each intent description
#      using TF-IDF vectors (or provide hooks for dense embeddings)
#   4. Return the highest-scoring intent
#
# def classify_intent(query: str) -> str
#
# Example:
queries = [
    "How much is my house worth?",       # → estimate_value
    "Show me apartments under $2000",    # → rent
    "I am looking to buy in Austin TX",  # → buy
]


# ---------------------------------------------------------------
# TIER 3 — ML FUNDAMENTALS
# ---------------------------------------------------------------

# PROBLEM 11: Naive Bayes Text Classifier
# Difficulty: Medium
# ---------------------------------------------------------------
# Train a Multinomial Naive Bayes classifier from scratch
# to classify listing descriptions by property type.
#
# Classes: ["house", "apartment", "condo", "townhouse"]
#
# Implement:
#   def train(corpus: list[str], labels: list[str]) -> model
#   def predict(model, text: str) -> str
#
# Use Laplace smoothing (alpha=1) for unseen words.
# Return log-probabilities to avoid underflow.


# PROBLEM 12: Precision, Recall, F1 — From Scratch
# Difficulty: Easy
# ---------------------------------------------------------------
# Your listing classifier produces predictions.
# Implement evaluation metrics for multi-class classification.
#
# def precision(y_true, y_pred, label) -> float
# def recall(y_true, y_pred, label) -> float
# def f1(y_true, y_pred, label) -> float
# def macro_f1(y_true, y_pred, labels) -> float
#
# Example:
y_true = ["house", "apartment", "house", "condo", "apartment"]
y_pred = ["house", "house",     "house", "condo", "apartment"]
# precision("apartment") = 1.0, recall("apartment") = 0.5, f1 = 0.667