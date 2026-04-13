"""
Day 012: Naive Bayes Classifier — Complete Implementation

A Multinomial Naive Bayes text classifier built from scratch.
Demonstrates Bayesian reasoning, log-space arithmetic, and Laplace smoothing.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for text.

    The core idea: for each class, learn a probability distribution over words.
    To classify a new document, ask "which class's word distribution makes this
    document most probable?" — adjusted by the prior probability of each class.

    All computation happens in log-space to avoid floating-point underflow.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Laplace smoothing parameter. alpha=1 is standard (add-one smoothing).
                   Higher values push word probabilities toward uniform.
                   Lower values trust the training counts more.
        """
        self.alpha = alpha
        # log P(class) for each class — the prior
        self.log_priors: Dict[str, float] = {}
        # log P(word | class) for each class — the likelihood table
        # Nested dict: class -> word -> log probability
        self.log_likelihoods: Dict[str, Dict[str, float]] = {}
        # Total word count per class (before smoothing) — needed for unseen words
        self.class_word_counts: Dict[str, int] = {}
        # The full vocabulary seen during training
        self.vocabulary: set = set()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Convert text to a list of lowercase word tokens.

        We strip punctuation and split on whitespace. This is intentionally simple —
        in production you'd use a proper tokenizer, handle contractions, maybe stem/lemmatize.
        But for learning the algorithm, simple tokenization keeps the focus on the math.
        """
        # Remove everything that isn't a letter, number, or space
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        # Split on whitespace and filter empty strings
        return [word for word in text.split() if word]

    def fit(self, documents: List[str], labels: List[str]) -> None:
        """
        Train the classifier by computing priors and word likelihoods.

        Training is just counting:
        1. Count documents per class → priors
        2. Count words per class → likelihoods (with smoothing)

        This is one of NB's big advantages: training is O(N * L) where N is the number
        of documents and L is average document length. No iterative optimization needed.
        """
        n_docs = len(documents)

        # --- Step 1: Compute class priors ---
        # P(class) = number of documents in class / total documents
        class_doc_counts: Counter = Counter(labels)

        # --- Step 2: Count words per class and build vocabulary ---
        # word_counts[class][word] = how many times word appears in documents of this class
        word_counts: Dict[str, Counter] = defaultdict(Counter)

        for doc, label in zip(documents, labels):
            tokens = self.tokenize(doc)
            word_counts[label].update(tokens)
            self.vocabulary.update(tokens)

        vocab_size = len(self.vocabulary)

        # --- Step 3: Convert counts to log probabilities ---
        for cls in class_doc_counts:
            # Prior: log P(class)
            self.log_priors[cls] = math.log(class_doc_counts[cls] / n_docs)

            # Total words in this class (before smoothing)
            total_words_in_class = sum(word_counts[cls].values())
            self.class_word_counts[cls] = total_words_in_class

            # Denominator for smoothed probability: total + alpha * vocab_size
            # This ensures all probabilities sum to 1 after smoothing
            denominator = total_words_in_class + self.alpha * vocab_size

            # Likelihood: log P(word | class) for each word in vocabulary
            self.log_likelihoods[cls] = {}
            for word in self.vocabulary:
                # Laplace smoothing: add alpha to every count
                count = word_counts[cls].get(word, 0) + self.alpha
                self.log_likelihoods[cls][word] = math.log(count / denominator)

    def predict_log_proba(self, text: str) -> Dict[str, float]:
        """
        Compute log P(class | document) for each class (up to a constant).

        log P(class | doc) ∝ log P(class) + Σ log P(word_i | class)

        We return unnormalized log-posteriors. The class with the highest value wins.
        Words not in the training vocabulary are simply ignored — they contribute
        equally (zero) to all classes, so they don't affect the ranking.
        """
        tokens = self.tokenize(text)
        vocab_size = len(self.vocabulary)

        scores: Dict[str, float] = {}
        for cls in self.log_priors:
            # Start with the prior
            score = self.log_priors[cls]

            for word in tokens:
                if word in self.log_likelihoods[cls]:
                    # Word was seen in training — use precomputed log probability
                    score += self.log_likelihoods[cls][word]
                elif word in self.vocabulary:
                    # Word is in vocabulary but wasn't counted for this class
                    # This shouldn't happen with our implementation (we compute for all vocab)
                    # but handle it defensively
                    denominator = self.class_word_counts[cls] + self.alpha * vocab_size
                    score += math.log(self.alpha / denominator)
                # Words not in vocabulary at all are ignored (contribute 0 to all classes)

            scores[cls] = score

        return scores

    def predict(self, text: str) -> str:
        """Classify a document by returning the class with highest log-posterior."""
        scores = self.predict_log_proba(text)
        # argmax: return the class with the highest score
        return max(scores, key=scores.get)

    def predict_batch(self, documents: List[str]) -> List[str]:
        """Classify multiple documents."""
        return [self.predict(doc) for doc in documents]


def evaluate(y_true: List[str], y_pred: List[str], positive_class: str = "spam") -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, precision, recall, F1.

    For binary classification, precision and recall depend on which class is "positive."
    In spam detection, "spam" is positive — we want to measure how well we catch spam
    (recall) without flagging legitimate mail (precision).

    - Precision = TP / (TP + FP) — "of messages we flagged as spam, how many really were?"
    - Recall = TP / (TP + FN) — "of actual spam messages, how many did we catch?"
    - F1 = harmonic mean of precision and recall — balances both concerns
    """
    tp = fp = fn = tn = 0
    for true, pred in zip(y_true, y_pred):
        if true == positive_class and pred == positive_class:
            tp += 1
        elif true != positive_class and pred == positive_class:
            fp += 1
        elif true == positive_class and pred != positive_class:
            fn += 1
        else:
            tn += 1

    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def train_test_split(
    documents: List[str], labels: List[str], test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Simple train/test split with deterministic shuffling.
    We implement this ourselves to avoid sklearn dependency.
    """
    import random
    random.seed(seed)

    indices = list(range(len(documents)))
    random.shuffle(indices)

    split_point = int(len(documents) * (1 - test_ratio))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    train_docs = [documents[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_docs = [documents[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_docs, test_docs, train_labels, test_labels


# =============================================================================
# Dataset: A curated set of SMS-style messages for spam classification
# =============================================================================

SPAM_MESSAGES = [
    "WINNER!! You have been selected for a cash prize! Claim now!",
    "Free entry in a weekly competition to win an iPad! Text WIN to 80888",
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim",
    "URGENT! Your account has been compromised. Verify your details immediately",
    "You have won a free ticket to the Bahamas! Call 1-800-FREE now",
    "Make money fast! Work from home and earn $5000/week guaranteed",
    "Limited time offer! Buy one get one free on all products",
    "You are a winner! Text YES to 12345 to claim your free prize",
    "Cheap pharmacy online! Best prices on medications delivered to your door",
    "Act now! This exclusive deal expires in 24 hours. Don't miss out!",
    "Credit card alert! You've been pre-approved for a $10,000 limit card",
    "Free ringtones! Download the hottest ringtones to your phone now",
    "Congratulations user! You have been selected for a special promotion",
    "Win cash prizes every week! Sign up now at our website for free",
    "URGENT: Your bank account needs verification. Click link to confirm",
    "Hot stock tip! Buy XYZ shares now before they triple in value",
    "You've been chosen to receive a free laptop! Just pay shipping",
    "Lose weight fast with our miracle diet pill! Order now",
    "Your phone number has won a prize! Call to collect your reward",
    "Free vacation package! All expenses paid trip. Reply YES to claim",
    "Double your investment guaranteed! Limited spots available act fast",
    "Exclusive VIP membership free for the first 100 callers only",
    "Your loan has been approved! Get cash deposited in 24 hours",
    "Free iPhone giveaway! Be one of the lucky winners today",
    "Alert: suspicious activity detected. Update your password at this link",
]

HAM_MESSAGES = [
    "Hey, are you coming to the meeting tomorrow at 3pm?",
    "Can you pick up some groceries on your way home? We need milk and bread",
    "Thanks for sending the report. I'll review it this afternoon",
    "Happy birthday! Hope you have a wonderful day",
    "The project deadline has been moved to next Friday",
    "Just finished the workout. Heading to shower then dinner",
    "Can we reschedule our lunch to Wednesday instead?",
    "I saw the movie last night. It was really good, you should watch it",
    "Don't forget we have a dentist appointment at 2pm",
    "The kids' school play is next Thursday evening at 7",
    "Running 10 minutes late to the office. Traffic is bad",
    "Did you see the game last night? What an incredible finish",
    "Let me know when you're free to discuss the budget proposal",
    "I'll pick up the dry cleaning after work today",
    "The restaurant reservation is confirmed for Saturday at 8pm",
    "Can you send me the slides from yesterday's presentation?",
    "Just landed safely. Will call you when I get to the hotel",
    "Remember to take your medication before bed tonight",
    "The plumber is coming between 9 and 11 tomorrow morning",
    "Great job on the quarterly review! The team did amazing work",
    "Want to go hiking this weekend if the weather is nice?",
    "I left my keys on the kitchen counter. Can you check?",
    "The wifi password for the guest network is sunshine2024",
    "Picking the kids up from soccer practice at 5:30",
    "Let's catch up over coffee sometime this week",
]


if __name__ == "__main__":
    print("=" * 70)
    print("Day 012: Naive Bayes Classifier — Spam Detection")
    print("=" * 70)

    # --- Prepare dataset ---
    documents = SPAM_MESSAGES + HAM_MESSAGES
    labels = ["spam"] * len(SPAM_MESSAGES) + ["ham"] * len(HAM_MESSAGES)

    print(f"\nDataset: {len(SPAM_MESSAGES)} spam + {len(HAM_MESSAGES)} ham = {len(documents)} total")

    # --- Train/test split ---
    train_docs, test_docs, train_labels, test_labels = train_test_split(
        documents, labels, test_ratio=0.2, seed=42
    )
    print(f"Train: {len(train_docs)} | Test: {len(test_docs)}")

    # --- Train the classifier ---
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(train_docs, train_labels)

    print(f"\nVocabulary size: {len(clf.vocabulary)} unique words")
    print(f"Classes: {list(clf.log_priors.keys())}")
    print(f"Priors: { {cls: f'{math.exp(lp):.2f}' for cls, lp in clf.log_priors.items()} }")

    # --- Show the most "spammy" and "hammy" words ---
    # Words with the biggest difference in log-likelihood between classes
    print("\n--- Most Discriminative Words ---")
    word_diffs = []
    for word in clf.vocabulary:
        spam_ll = clf.log_likelihoods.get("spam", {}).get(word, -10)
        ham_ll = clf.log_likelihoods.get("ham", {}).get(word, -10)
        word_diffs.append((word, spam_ll - ham_ll))

    word_diffs.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 spam indicators:", [w for w, _ in word_diffs[:10]])
    print("Top 10 ham indicators: ", [w for w, _ in word_diffs[-10:]])

    # --- Classify test set ---
    print("\n--- Test Set Predictions ---")
    test_preds = clf.predict_batch(test_docs)

    for doc, true, pred in zip(test_docs, test_labels, test_preds):
        status = "OK" if true == pred else "WRONG"
        short_doc = doc[:50] + "..." if len(doc) > 50 else doc
        print(f"  [{status}] True={true:4s} Pred={pred:4s} | {short_doc}")

    # --- Evaluate ---
    print("\n--- Evaluation Metrics ---")
    metrics = evaluate(test_labels, test_preds, positive_class="spam")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:>10s}: {value:.4f}")
        else:
            print(f"  {name:>10s}: {value}")

    # --- Detailed walkthrough of a single prediction ---
    print("\n--- Detailed Prediction Walkthrough ---")
    test_msg = "Congratulations! You've won a free trip. Call now to claim your prize!"
    print(f"Message: '{test_msg}'")

    tokens = clf.tokenize(test_msg)
    print(f"Tokens: {tokens}")

    scores = clf.predict_log_proba(test_msg)
    print(f"\nLog-posteriors (unnormalized):")
    for cls, score in scores.items():
        print(f"  P({cls} | message) ∝ exp({score:.2f})")

    # Show per-word contributions
    print(f"\nPer-word log-likelihood contributions:")
    for cls in clf.log_priors:
        print(f"\n  Class: {cls} (prior: {clf.log_priors[cls]:.3f})")
        for word in tokens:
            if word in clf.log_likelihoods[cls]:
                ll = clf.log_likelihoods[cls][word]
                print(f"    '{word}': {ll:.3f}")

    prediction = clf.predict(test_msg)
    print(f"\nFinal prediction: {prediction}")

    # --- Effect of smoothing parameter ---
    print("\n--- Effect of Smoothing Parameter (alpha) ---")
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        clf_a = NaiveBayesClassifier(alpha=alpha)
        clf_a.fit(train_docs, train_labels)
        preds_a = clf_a.predict_batch(test_docs)
        metrics_a = evaluate(test_labels, preds_a, positive_class="spam")
        print(f"  alpha={alpha:<5.2f} -> accuracy={metrics_a['accuracy']:.4f}, f1={metrics_a['f1']:.4f}")

    print("\nDone!")
