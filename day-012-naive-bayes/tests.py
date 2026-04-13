"""
Day 012: Naive Bayes Classifier — Test Suite

Run with: python3 -m pytest tests.py -v
      or: python3 tests.py
"""

import math
import unittest
from my_solution import NaiveBayesClassifier, evaluate, train_test_split


class TestTokenize(unittest.TestCase):
    """Test the tokenization step."""

    def test_basic_tokenization(self):
        tokens = NaiveBayesClassifier.tokenize("Hello World")
        self.assertEqual(tokens, ["hello", "world"])

    def test_punctuation_removal(self):
        tokens = NaiveBayesClassifier.tokenize("Win!!! FREE prize, now.")
        self.assertIn("win", tokens)
        self.assertIn("free", tokens)
        self.assertIn("prize", tokens)
        # No punctuation should remain
        for t in tokens:
            self.assertTrue(t.isalnum(), f"Token '{t}' contains non-alphanumeric chars")

    def test_empty_string(self):
        tokens = NaiveBayesClassifier.tokenize("")
        self.assertEqual(tokens, [])


class TestFit(unittest.TestCase):
    """Test the training step."""

    def setUp(self):
        self.clf = NaiveBayesClassifier(alpha=1.0)
        self.docs = ["free money now", "free prize win", "hello friend meeting"]
        self.labels = ["spam", "spam", "ham"]
        self.clf.fit(self.docs, self.labels)

    def test_priors_computed(self):
        """Priors should reflect class frequencies."""
        self.assertIn("spam", self.clf.log_priors)
        self.assertIn("ham", self.clf.log_priors)
        # 2 spam out of 3 docs -> P(spam) = 2/3
        self.assertAlmostEqual(math.exp(self.clf.log_priors["spam"]), 2 / 3, places=5)
        self.assertAlmostEqual(math.exp(self.clf.log_priors["ham"]), 1 / 3, places=5)

    def test_vocabulary_built(self):
        """Vocabulary should contain all unique words from training."""
        expected = {"free", "money", "now", "prize", "win", "hello", "friend", "meeting"}
        self.assertEqual(self.clf.vocabulary, expected)

    def test_likelihoods_computed(self):
        """Every word in vocab should have a likelihood for every class."""
        for cls in ["spam", "ham"]:
            for word in self.clf.vocabulary:
                self.assertIn(word, self.clf.log_likelihoods[cls])

    def test_smoothing_prevents_zero(self):
        """Words not seen in a class should still have non-zero probability due to smoothing."""
        # "meeting" only appears in ham, but should have non-zero probability in spam
        spam_ll = self.clf.log_likelihoods["spam"]["meeting"]
        self.assertTrue(math.isfinite(spam_ll))
        self.assertLess(spam_ll, 0)  # log of a probability is negative


class TestPredict(unittest.TestCase):
    """Test classification predictions."""

    def setUp(self):
        self.clf = NaiveBayesClassifier(alpha=1.0)
        spam = [
            "free money win prize claim",
            "win free cash now urgent",
            "claim your prize winner free",
            "free offer limited time only",
        ]
        ham = [
            "meeting tomorrow at the office",
            "can you pick up groceries",
            "the project deadline is friday",
            "great job on the report today",
        ]
        self.clf.fit(spam + ham, ["spam"] * 4 + ["ham"] * 4)

    def test_classifies_obvious_spam(self):
        self.assertEqual(self.clf.predict("free prize winner claim now"), "spam")

    def test_classifies_obvious_ham(self):
        self.assertEqual(self.clf.predict("meeting at the office tomorrow"), "ham")

    def test_predict_batch(self):
        docs = ["free money", "office meeting"]
        preds = self.clf.predict_batch(docs)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], "spam")
        self.assertEqual(preds[1], "ham")

    def test_log_proba_returns_all_classes(self):
        scores = self.clf.predict_log_proba("test message")
        self.assertIn("spam", scores)
        self.assertIn("ham", scores)


class TestEvaluate(unittest.TestCase):
    """Test the evaluation metrics."""

    def test_perfect_predictions(self):
        y_true = ["spam", "spam", "ham", "ham"]
        y_pred = ["spam", "spam", "ham", "ham"]
        m = evaluate(y_true, y_pred, positive_class="spam")
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_all_wrong(self):
        y_true = ["spam", "spam", "ham", "ham"]
        y_pred = ["ham", "ham", "spam", "spam"]
        m = evaluate(y_true, y_pred, positive_class="spam")
        self.assertAlmostEqual(m["accuracy"], 0.0)
        self.assertAlmostEqual(m["precision"], 0.0)
        self.assertAlmostEqual(m["recall"], 0.0)

    def test_mixed_results(self):
        y_true = ["spam", "spam", "ham", "ham"]
        y_pred = ["spam", "ham", "ham", "spam"]
        m = evaluate(y_true, y_pred, positive_class="spam")
        # TP=1, FP=1, FN=1, TN=1
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)


class TestEndToEnd(unittest.TestCase):
    """Full pipeline test with the provided dataset."""

    def test_achieves_reasonable_accuracy(self):
        """The classifier should achieve >70% accuracy on the SMS dataset."""
        from my_solution import SPAM_MESSAGES, HAM_MESSAGES

        docs = SPAM_MESSAGES + HAM_MESSAGES
        labels = ["spam"] * len(SPAM_MESSAGES) + ["ham"] * len(HAM_MESSAGES)

        train_docs, test_docs, train_labels, test_labels = train_test_split(
            docs, labels, test_ratio=0.2, seed=42
        )

        clf = NaiveBayesClassifier(alpha=1.0)
        clf.fit(train_docs, train_labels)
        preds = clf.predict_batch(test_docs)
        metrics = evaluate(test_labels, preds)

        self.assertGreater(metrics["accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
