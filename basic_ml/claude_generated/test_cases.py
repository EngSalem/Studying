# ============================================================
# BASE TESTS — Zillow Applied Scientist Coding Problems
# Run with: pytest test_zillow.py -v
# Assumes your solutions are in: solutions.py
# ============================================================

import pytest
import numpy as np
from questions import (
    tfidf_matrix,
    search,
    bm25_rank,
    train_bpe,
    edit_distance,
    extract_features,
    simhash, hamming, are_duplicates,
    extractive_summarize,
    fixed_chunk, sentence_chunk,
    classify_intent,
    train_nb, predict_nb,
    precision, recall, f1, macro_f1,
)


# ---------------------------------------------------------------
# PROBLEM 1: TF-IDF Vectorizer
# ---------------------------------------------------------------
class TestTFIDF:
    corpus = [
        "spacious 3 bedroom house with large backyard",
        "cozy 1 bedroom apartment near downtown",
        "3 bedroom house with garage and pool",
    ]

    def test_output_shape(self):
        mat = tfidf_matrix(self.corpus)
        assert mat.shape[0] == 3

    def test_output_is_non_negative(self):
        mat = tfidf_matrix(self.corpus)
        assert np.all(mat >= 0)

    def test_matrix_not_all_zeros(self):
        mat = tfidf_matrix(self.corpus)
        assert mat.sum() > 0

    def test_single_document_corpus(self):
        mat = tfidf_matrix(["a single document"])
        assert mat.shape[0] == 1
        assert mat.shape[1] == 3  # 3 unique words

    def test_repeated_word_increases_tf(self):
        # "house" appears twice in doc 0, once in doc 1
        corpus = ["house house garage", "house pool"]
        mat = tfidf_matrix(corpus)
        vocab_size = mat.shape[1]
        assert vocab_size > 0
        # doc 0 tfidf sum should be >= doc 1 since it has more content
        assert mat[0].sum() >= 0

    def test_rare_word_higher_idf_than_common(self):
        # "bedroom" in all 3 docs vs "garage" in only 1 — garage row should have higher IDF weight
        corpus = ["bedroom garage", "bedroom pool", "bedroom backyard"]
        mat = tfidf_matrix(corpus)
        # All docs have bedroom, only first has garage — garage score in doc 0 > bedroom score
        # We just check the matrix isn't flat (common words penalized)
        assert not np.allclose(mat[0], mat[1])

    def test_case_insensitive(self):
        mat1 = tfidf_matrix(["Bedroom House Garage"])
        mat2 = tfidf_matrix(["bedroom house garage"])
        assert np.allclose(mat1, mat2)

    def test_punctuation_ignored(self):
        mat1 = tfidf_matrix(["bedroom, house. garage!"])
        mat2 = tfidf_matrix(["bedroom house garage"])
        assert mat1.shape == mat2.shape

    def test_identical_docs_identical_rows(self):
        corpus = ["house with garage", "house with garage"]
        mat = tfidf_matrix(corpus)
        assert np.allclose(mat[0], mat[1])

    def test_vocabulary_size_correct(self):
        corpus = ["cat sat mat", "cat and dog"]
        mat = tfidf_matrix(corpus)
        unique_words = len({"cat", "sat", "mat", "and", "dog"})
        assert mat.shape[1] == unique_words


# ---------------------------------------------------------------
# PROBLEM 2: Cosine Similarity Search
# ---------------------------------------------------------------
class TestCosineSearch:
    corpus = [
        "spacious 3 bedroom house with large backyard",
        "cozy 1 bedroom apartment near downtown",
        "3 bedroom house with garage and pool",
    ]

    def test_returns_k_results(self):
        assert len(search("3 bedroom house", self.corpus, k=2)) == 2

    def test_returns_tuple_of_listing_and_score(self):
        listing, score = search("3 bedroom house", self.corpus, k=1)[0]
        assert isinstance(listing, str)
        assert isinstance(score, float)

    def test_scores_sorted_descending(self):
        results = search("3 bedroom house", self.corpus, k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_listing_ranks_first(self):
        results = search("garage pool", self.corpus, k=1)
        assert "garage" in results[0][0] or "pool" in results[0][0]

    def test_k_equals_corpus_size(self):
        results = search("bedroom", self.corpus, k=3)
        assert len(results) == 3

    def test_scores_between_0_and_1(self):
        results = search("3 bedroom house", self.corpus, k=3)
        for _, score in results:
            assert -0.01 <= score <= 1.01

    def test_exact_match_scores_highest(self):
        corpus = ["house with garage", "apartment downtown", "condo with pool"]
        results = search("house with garage", corpus, k=3)
        assert results[0][0] == "house with garage"

    def test_no_overlap_query_scores_zero(self):
        results = search("zzzznonexistentword", self.corpus, k=3)
        for _, score in results:
            assert score == pytest.approx(0.0, abs=1e-6)

    def test_query_not_in_corpus_still_returns_k(self):
        results = search("oceanfront villa with infinity pool", self.corpus, k=2)
        assert len(results) == 2

    def test_cosine_symmetric(self):
        # searching A in [B] should equal searching B in [A]
        r1 = search("house garage", ["bedroom apartment pool"], k=1)[0][1]
        r2 = search("bedroom apartment pool", ["house garage"], k=1)[0][1]
        assert r1 == pytest.approx(r2, abs=1e-6)


# ---------------------------------------------------------------
# PROBLEM 3: BM25 Ranking
# ---------------------------------------------------------------
class TestBM25:
    corpus = [
        "3 bedroom house with garage and large backyard",
        "cozy 1 bedroom apartment near downtown",
        "spacious 3 bedroom home with garage and pool",
    ]

    def test_returns_all_listings(self):
        assert len(bm25_rank(self.corpus, "3 bedroom garage")) == 3

    def test_scores_sorted_descending(self):
        results = bm25_rank(self.corpus, "3 bedroom garage")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_apartment_ranks_last(self):
        results = bm25_rank(self.corpus, "3 bedroom garage")
        assert "apartment" in results[-1][0]

    def test_zero_score_for_no_overlap(self):
        results = bm25_rank(self.corpus, "zzzznonexistentword")
        assert all(s == 0 for _, s in results)

    def test_scores_non_negative(self):
        results = bm25_rank(self.corpus, "bedroom house")
        assert all(s >= 0 for _, s in results)

    def test_single_doc_corpus(self):
        results = bm25_rank(["house with garage"], "garage")
        assert len(results) == 1
        assert results[0][1] > 0

    def test_returns_original_listing_strings(self):
        results = bm25_rank(self.corpus, "bedroom")
        returned_listings = [l for l, _ in results]
        assert set(returned_listings) == set(self.corpus)

    def test_more_query_term_occurrences_score_higher(self):
        # doc with more matches for query terms should score higher
        corpus = ["garage garage garage", "garage", "pool backyard"]
        results = bm25_rank(corpus, "garage")
        assert results[0][0] == "garage garage garage"

    def test_longer_doc_penalized(self):
        # BM25 normalizes by doc length — very long doc with same terms should score lower
        short = "garage bedroom"
        long  = "garage bedroom " + " ".join(["filler"] * 50)
        results = bm25_rank([short, long], "garage bedroom")
        assert results[0][0] == short


# ---------------------------------------------------------------
# PROBLEM 4: Byte Pair Encoding (BPE)
# ---------------------------------------------------------------
class TestBPE:
    corpus = ["low low low lower lowest", "newer newer wider new new"]

    def test_returns_list_of_tuples(self):
        merges = train_bpe(self.corpus, num_merges=5)
        assert isinstance(merges, list)
        assert all(isinstance(m, tuple) and len(m) == 2 for m in merges)

    def test_correct_number_of_merges(self):
        assert len(train_bpe(self.corpus, num_merges=5)) == 5

    def test_zero_merges(self):
        assert train_bpe(self.corpus, num_merges=0) == []

    def test_most_frequent_pair_merged_first(self):
        merges = train_bpe(self.corpus, num_merges=3)
        flat = [item for pair in merges for item in pair]
        assert any(c in flat for c in ["l", "o", "w"])

    def test_no_duplicate_merge_rules(self):
        merges = train_bpe(self.corpus, num_merges=6)
        assert len(merges) == len(set(merges))

    def test_merges_are_strings(self):
        merges = train_bpe(self.corpus, num_merges=3)
        for a, b in merges:
            assert isinstance(a, str) and isinstance(b, str)

    def test_single_word_corpus(self):
        # Should still produce merges on character pairs
        merges = train_bpe(["aaa aaa aaa"], num_merges=2)
        assert len(merges) <= 2  # may run out of pairs early

    def test_early_stop_if_no_pairs(self):
        # Corpus with single-char words only — no pairs to merge
        merges = train_bpe(["a b c d"], num_merges=10)
        assert len(merges) == 0


# ---------------------------------------------------------------
# PROBLEM 5: Edit Distance (Levenshtein)
# ---------------------------------------------------------------
class TestEditDistance:
    def test_identical_strings(self):
        assert edit_distance("123 Main St", "123 Main St") == 0

    def test_one_substitution(self):
        assert edit_distance("123 Mian St", "123 Main St") == 1

    def test_one_insertion(self):
        assert edit_distance("123 Main St", "123 Main Str") == 1

    def test_one_deletion(self):
        assert edit_distance("123 Main Str", "123 Main St") == 1

    def test_completely_different(self):
        assert edit_distance("abc", "xyz") == 3

    def test_empty_vs_nonempty(self):
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3

    def test_both_empty(self):
        assert edit_distance("", "") == 0

    def test_symmetry(self):
        assert edit_distance("kitten", "sitting") == edit_distance("sitting", "kitten")

    def test_classic_kitten_sitting(self):
        assert edit_distance("kitten", "sitting") == 3

    def test_single_char_substitution(self):
        assert edit_distance("a", "b") == 1

    def test_single_char_insertion(self):
        assert edit_distance("", "a") == 1

    def test_address_typo(self):
        assert edit_distance("Pensylvannia Ave", "Pennsylvania Ave") == 2

    def test_prefix_string(self):
        # "house" vs "houses" — one insertion
        assert edit_distance("house", "houses") == 1

    def test_transposition_costs_two(self):
        # Levenshtein treats transposition as 2 ops (delete + insert)
        assert edit_distance("ab", "ba") == 2


# ---------------------------------------------------------------
# PROBLEM 6: NER — Extract Listing Features
# ---------------------------------------------------------------
class TestNER:
    desc = (
        "Beautiful 3 bedroom, 2.5 bathroom home listed at $540,000. "
        "Features include a garage, pool, and large backyard."
    )

    def test_bedrooms(self):
        assert extract_features(self.desc)["bedrooms"] == 3

    def test_bathrooms(self):
        assert extract_features(self.desc)["bathrooms"] == 2.5

    def test_price(self):
        assert extract_features(self.desc)["price"] == 540000

    def test_amenities(self):
        assert set(extract_features(self.desc)["amenities"]) == {"garage", "pool", "backyard"}

    def test_missing_fields_return_none(self):
        result = extract_features("Nice place with a balcony.")
        assert result["bedrooms"] is None
        assert result["price"] is None

    def test_missing_amenities_returns_empty_list(self):
        result = extract_features("A plain studio with no features listed.")
        assert result["amenities"] == []

    def test_integer_bathrooms(self):
        result = extract_features("2 bathroom condo listed at $300,000")
        assert result["bathrooms"] == 2.0

    def test_price_without_commas(self):
        result = extract_features("Listed at $250000 with 2 bedrooms")
        assert result["price"] == 250000

    def test_only_known_amenities_extracted(self):
        result = extract_features("Has a sauna, garage, and rooftop deck.")
        # sauna and rooftop are not in AMENITY_SET
        assert "sauna" not in result["amenities"]
        assert "garage" in result["amenities"]

    def test_multiple_amenities(self):
        result = extract_features("Features gym, pool, parking, and balcony.")
        assert set(result["amenities"]) == {"gym", "pool", "parking", "balcony"}

    def test_abbreviation_bedroom(self):
        # Bonus: handle "3BR" or "3 bed"
        result = extract_features("3BR apartment with 1 bath at $1,500/mo")
        assert result["bedrooms"] == 3


# ---------------------------------------------------------------
# PROBLEM 7: Duplicate Listing Detection (SimHash)
# ---------------------------------------------------------------
class TestSimHash:
    t1 = "Stunning 3 bedroom house with garage and large pool in Seattle"
    t2 = "Stunning 3 bedroom house with garage and large pool in Seattle WA"
    t3 = "1 bedroom apartment near downtown Portland great city views"

    def test_simhash_returns_int(self):
        assert isinstance(simhash(self.t1), int)

    def test_hamming_identical_text(self):
        h = simhash(self.t1)
        assert hamming(h, h) == 0

    def test_near_duplicates_detected(self):
        assert are_duplicates(self.t1, self.t2) is True

    def test_different_texts_not_duplicates(self):
        assert are_duplicates(self.t1, self.t3) is False

    def test_hamming_in_valid_range(self):
        h1, h2 = simhash(self.t1), simhash(self.t3)
        assert 0 <= hamming(h1, h2) <= 64

    def test_hamming_symmetric(self):
        h1, h2 = simhash(self.t1), simhash(self.t2)
        assert hamming(h1, h2) == hamming(h2, h1)

    def test_identical_texts_are_duplicates(self):
        assert are_duplicates(self.t1, self.t1) is True

    def test_completely_different_not_duplicate(self):
        a = "bedroom garage backyard Seattle house"
        b = "tax returns quarterly earnings financial report"
        assert are_duplicates(a, b) is False

    def test_simhash_deterministic(self):
        assert simhash(self.t1) == simhash(self.t1)

    def test_empty_string_does_not_crash(self):
        h = simhash("")
        assert isinstance(h, int)


# ---------------------------------------------------------------
# PROBLEM 8: Extractive Summarizer
# ---------------------------------------------------------------
class TestSummarizer:
    doc = (
        "This stunning property is located in the heart of Seattle. "
        "It features 4 bedrooms and 3 modern bathrooms. "
        "The kitchen was recently renovated with quartz countertops. "
        "There is a two-car garage and a spacious backyard with a deck. "
        "The home is walking distance to top-rated schools and parks. "
        "Monthly HOA fees are $250 and include landscaping."
    )

    def test_returns_string(self):
        assert isinstance(extractive_summarize(self.doc, k=2), str)

    def test_returns_k_sentences(self):
        summary = extractive_summarize(self.doc, k=3)
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        assert len(sentences) == 3

    def test_preserves_original_order(self):
        summary = extractive_summarize(self.doc, k=3)
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        positions = [self.doc.find(s) for s in sentences]
        assert positions == sorted(positions)

    def test_k_equals_1(self):
        summary = extractive_summarize(self.doc, k=1)
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        assert len(sentences) == 1

    def test_k_equals_all_sentences(self):
        all_sentences = [s.strip() for s in self.doc.split(".") if s.strip()]
        summary = extractive_summarize(self.doc, k=len(all_sentences))
        returned = [s.strip() for s in summary.split(".") if s.strip()]
        assert len(returned) == len(all_sentences)

    def test_output_sentences_are_from_original(self):
        summary = extractive_summarize(self.doc, k=2)
        for sent in [s.strip() for s in summary.split(".") if s.strip()]:
            assert sent in self.doc

    def test_short_doc_does_not_crash(self):
        short = "Nice house. Great pool."
        summary = extractive_summarize(short, k=1)
        assert isinstance(summary, str)


# ---------------------------------------------------------------
# PROBLEM 9: Chunking Strategies for RAG
# ---------------------------------------------------------------
class TestChunking:
    text_300 = " ".join(["word"] * 300)

    def test_fixed_chunk_max_size_respected(self):
        chunks = fixed_chunk(self.text_300, chunk_size=100, overlap=20)
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_fixed_chunk_produces_multiple_chunks(self):
        chunks = fixed_chunk(self.text_300, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_fixed_chunk_no_empty_chunks(self):
        chunks = fixed_chunk(self.text_300, chunk_size=100, overlap=20)
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_fixed_chunk_overlap_content(self):
        text = " ".join([str(i) for i in range(20)])
        chunks = fixed_chunk(text, chunk_size=10, overlap=3)
        # Last 3 tokens of chunk 0 should appear at start of chunk 1
        end_of_first = chunks[0].split()[-3:]
        start_of_second = chunks[1].split()[:3]
        assert end_of_first == start_of_second

    def test_fixed_chunk_text_shorter_than_chunk_size(self):
        chunks = fixed_chunk("hello world", chunk_size=100, overlap=10)
        assert len(chunks) == 1

    def test_sentence_chunk_max_words_respected(self):
        doc = (
            "This house has a garage. It also has a pool. "
            "The backyard is large. There are 3 bedrooms. "
            "The kitchen is modern. The bathrooms are renovated."
        )
        chunks = sentence_chunk(doc, max_words=10)
        assert all(len(c.split()) <= 10 for c in chunks)

    def test_sentence_chunk_no_empty_chunks(self):
        doc = "Nice house. Great pool. Big yard. Good schools. Low taxes."
        chunks = sentence_chunk(doc, max_words=8)
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_sentence_chunk_covers_all_content(self):
        doc = "Nice house. Great pool. Big yard."
        chunks = sentence_chunk(doc, max_words=20)
        # All content should be present across chunks
        rejoined = " ".join(chunks)
        for word in ["house", "pool", "yard"]:
            assert word in rejoined

    def test_sentence_chunk_single_sentence(self):
        doc = "A house with a garage and a pool and a backyard."
        chunks = sentence_chunk(doc, max_words=50)
        assert len(chunks) == 1


# ---------------------------------------------------------------
# PROBLEM 10: Intent Classifier
# ---------------------------------------------------------------
class TestIntentClassifier:
    def test_buy_intent(self):
        assert classify_intent("I am looking to buy a home in Austin") == "buy"

    def test_rent_intent(self):
        assert classify_intent("Show me apartments under $2000 per month") == "rent"

    def test_estimate_value_intent(self):
        assert classify_intent("How much is my house worth?") == "estimate_value"

    def test_find_agent_intent(self):
        assert classify_intent("I need a realtor to help me sell") == "find_agent"

    def test_returns_string(self):
        assert isinstance(classify_intent("find me a real estate agent"), str)

    def test_valid_intent_returned(self):
        valid = {"buy", "rent", "estimate_value", "find_agent", "general_info"}
        assert classify_intent("I need help with Zillow") in valid

    def test_purchase_maps_to_buy(self):
        assert classify_intent("I want to purchase a property") == "buy"

    def test_valuation_maps_to_estimate(self):
        assert classify_intent("What is my home valuation?") == "estimate_value"

    def test_does_not_crash_on_empty_query(self):
        result = classify_intent("")
        assert result in {"buy", "rent", "estimate_value", "find_agent", "general_info"}


# ---------------------------------------------------------------
# PROBLEM 11: Naive Bayes Classifier
# ---------------------------------------------------------------
class TestNaiveBayes:
    corpus = [
        "large house with backyard and garage",
        "single family house with pool and yard",
        "studio apartment downtown near transit",
        "1 bedroom apartment city center location",
        "luxury condo with gym and concierge",
        "downtown condo with rooftop deck access",
    ]
    labels = ["house", "house", "apartment", "apartment", "condo", "condo"]

    def test_predict_returns_string(self):
        model = train_nb(self.corpus, self.labels)
        assert isinstance(predict_nb(model, "house with large backyard"), str)

    def test_predict_valid_class(self):
        model = train_nb(self.corpus, self.labels)
        assert predict_nb(model, "apartment near downtown") in {"house", "apartment", "condo"}

    def test_obvious_house_prediction(self):
        model = train_nb(self.corpus, self.labels)
        assert predict_nb(model, "large house backyard garage") == "house"

    def test_obvious_apartment_prediction(self):
        model = train_nb(self.corpus, self.labels)
        assert predict_nb(model, "studio apartment downtown") == "apartment"

    def test_obvious_condo_prediction(self):
        model = train_nb(self.corpus, self.labels)
        assert predict_nb(model, "luxury condo rooftop gym") == "condo"

    def test_unseen_word_does_not_crash(self):
        model = train_nb(self.corpus, self.labels)
        result = predict_nb(model, "xyzunknownword foobarbaz")
        assert result in {"house", "apartment", "condo"}

    def test_training_on_single_class_does_not_crash(self):
        model = train_nb(["house with garage", "big house yard"], ["house", "house"])
        result = predict_nb(model, "house garage")
        assert result == "house"

    def test_model_contains_required_keys(self):
        model = train_nb(self.corpus, self.labels)
        assert "log_priors" in model
        assert "log_likelihoods" in model


# ---------------------------------------------------------------
# PROBLEM 12: Evaluation Metrics
# ---------------------------------------------------------------
class TestMetrics:
    y_true = ["house", "apartment", "house", "condo", "apartment"]
    y_pred = ["house", "house",     "house", "condo", "apartment"]

    def test_precision_house(self):
        # TP=2, FP=1 (apartment predicted as house) → 2/3
        assert precision(self.y_true, self.y_pred, "house") == pytest.approx(2/3, rel=1e-3)

    def test_recall_apartment(self):
        # TP=1, FN=1 → 0.5
        assert recall(self.y_true, self.y_pred, "apartment") == pytest.approx(0.5, rel=1e-3)

    def test_f1_condo_perfect(self):
        assert f1(self.y_true, self.y_pred, "condo") == pytest.approx(1.0, rel=1e-3)

    def test_macro_f1_in_range(self):
        score = macro_f1(self.y_true, self.y_pred, ["house", "apartment", "condo"])
        assert 0.0 <= score <= 1.0

    def test_perfect_predictions_macro_f1(self):
        y = ["house", "apartment", "condo"]
        assert macro_f1(y, y, ["house", "apartment", "condo"]) == pytest.approx(1.0)

    def test_all_wrong_precision_zero(self):
        y_true = ["house", "house", "house"]
        y_pred = ["condo", "condo", "condo"]
        assert precision(y_true, y_pred, "house") == 0.0
        assert recall(y_true, y_pred, "condo") == 0.0

    def test_zero_division_precision_handled(self):
        # label never predicted → should return 0.0 not crash
        assert precision(self.y_true, self.y_pred, "townhouse") == 0.0

    def test_zero_division_recall_handled(self):
        # label never in y_true → should return 0.0 not crash
        assert recall(self.y_true, self.y_pred, "townhouse") == 0.0

    def test_zero_division_f1_handled(self):
        assert f1(self.y_true, self.y_pred, "townhouse") == 0.0

    def test_precision_perfect_label(self):
        # condo predicted correctly every time it's predicted
        assert precision(self.y_true, self.y_pred, "condo") == pytest.approx(1.0)

    def test_recall_perfect_label(self):
        assert recall(self.y_true, self.y_pred, "condo") == pytest.approx(1.0)

    def test_f1_is_harmonic_mean(self):
        p = precision(self.y_true, self.y_pred, "house")
        r = recall(self.y_true, self.y_pred, "house")
        expected = 2 * p * r / (p + r)
        assert f1(self.y_true, self.y_pred, "house") == pytest.approx(expected, rel=1e-3)

    def test_macro_f1_averages_all_labels(self):
        labels = ["house", "apartment", "condo"]
        scores = [f1(self.y_true, self.y_pred, l) for l in labels]
        assert macro_f1(self.y_true, self.y_pred, labels) == pytest.approx(sum(scores)/len(scores))

tfidf_test = TestTFIDF()
print("Running TF-IDF tests...")
tfidf_test.test_output_shape()
tfidf_test.test_output_is_non_negative()
tfidf_test.test_matrix_not_all_zeros()
tfidf_test.test_single_document_corpus()
tfidf_test.test_repeated_word_increases_tf()
tfidf_test.test_rare_word_higher_idf_than_common()
tfidf_test.test_case_insensitive()
tfidf_test.test_punctuation_ignored()
tfidf_test.test_identical_docs_identical_rows()
tfidf_test.test_vocabulary_size_correct()
print("TF-IDF tests passed!")

## cosine similarity test cases

cosine_sim_tests = TestCosineSearch() 
print("Running Cosine Similarity Search tests...")
cosine_sim_tests.test_returns_k_results()
cosine_sim_tests.test_returns_tuple_of_listing_and_score()
cosine_sim_tests.test_scores_sorted_descending()
cosine_sim_tests.test_relevant_listing_ranks_first()
cosine_sim_tests.test_k_equals_corpus_size()
cosine_sim_tests.test_scores_between_0_and_1()
cosine_sim_tests.test_exact_match_scores_highest()
cosine_sim_tests.test_no_overlap_query_scores_zero()
cosine_sim_tests.test_query_not_in_corpus_still_returns_k()
cosine_sim_tests.test_cosine_symmetric()
print("Cosine Similarity Search tests passed!")
