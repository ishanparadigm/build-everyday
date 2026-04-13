"""
Day 012: Naive Bayes Classifier — Your Implementation

Build a Multinomial Naive Bayes text classifier from scratch.
No sklearn or ML libraries — just Python, math, and probability theory.

Hint: The entire algorithm boils down to counting words per class,
then picking the class whose word distribution best explains a new document.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for text.

    Key formulas to implement:
    - Prior: P(class) = docs_in_class / total_docs
    - Likelihood: P(word|class) = (count(word,class) + alpha) / (total_words_in_class + alpha * vocab_size)
    - Posterior: log P(class|doc) ∝ log P(class) + Σ log P(word_i|class)
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Laplace smoothing parameter.
        """
        self.alpha = alpha
        self.log_priors: Dict[str, float] = {}
        self.log_likelihoods: Dict[str, Dict[str, float]] = {}
        self.class_word_counts: Dict[str, int] = {}
        self.vocabulary: set = set()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Convert text to lowercase word tokens, removing punctuation.

        Hint: Use regex to strip non-alphanumeric characters, then split on whitespace.
        """
        raise NotImplementedError("TODO: implement tokenization")

    def fit(self, documents: List[str], labels: List[str]) -> None:
        """
        Train the classifier on labeled documents.

        Steps:
        1. Count documents per class to compute priors
        2. Tokenize each document and count words per class
        3. Build the vocabulary (set of all unique words)
        4. Compute log P(class) for each class
        5. Compute log P(word|class) with Laplace smoothing for each word and class

        Hint: Store everything as log probabilities to avoid underflow.
        Hint: The smoothed denominator is: total_words_in_class + alpha * vocab_size
        """
        raise NotImplementedError("TODO: implement training")

    def predict_log_proba(self, text: str) -> Dict[str, float]:
        """
        Compute unnormalized log-posterior for each class.

        For each class:
          score = log P(class) + sum of log P(word|class) for each word in text

        Hint: Skip words not in the vocabulary — they contribute equally to all classes.
        """
        raise NotImplementedError("TODO: implement log-posterior computation")

    def predict(self, text: str) -> str:
        """Return the class with the highest log-posterior."""
        raise NotImplementedError("TODO: implement prediction")

    def predict_batch(self, documents: List[str]) -> List[str]:
        """Classify multiple documents."""
        raise NotImplementedError("TODO: implement batch prediction")


def evaluate(y_true: List[str], y_pred: List[str], positive_class: str = "spam") -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, and F1 score.

    Hint:
    - TP = true positive (predicted spam, actually spam)
    - FP = false positive (predicted spam, actually ham)
    - FN = false negative (predicted ham, actually spam)
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 = 2 * P * R / (P + R)
    """
    raise NotImplementedError("TODO: implement evaluation metrics")


def train_test_split(
    documents: List[str], labels: List[str], test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split data into train and test sets with deterministic shuffling."""
    import random
    random.seed(seed)

    indices = list(range(len(documents)))
    random.shuffle(indices)

    split_point = int(len(documents) * (1 - test_ratio))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    return (
        [documents[i] for i in train_idx],
        [documents[i] for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx],
    )


# Dataset for testing
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
    print("Naive Bayes Classifier — Test your implementation")
    print("=" * 50)

    documents = SPAM_MESSAGES + HAM_MESSAGES
    labels = ["spam"] * len(SPAM_MESSAGES) + ["ham"] * len(HAM_MESSAGES)

    train_docs, test_docs, train_labels, test_labels = train_test_split(
        documents, labels, test_ratio=0.2, seed=42
    )

    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(train_docs, train_labels)

    print(f"Vocabulary size: {len(clf.vocabulary)}")
    print(f"Classes: {list(clf.log_priors.keys())}")

    # Test on a few examples
    test_messages = [
        "You've won a free prize! Call now!",
        "Can you pick me up from work at 5?",
        "URGENT: verify your account details immediately",
        "Let's grab dinner tonight at that new restaurant",
    ]

    for msg in test_messages:
        pred = clf.predict(msg)
        print(f"  '{msg[:50]}...' -> {pred}")

    # Evaluate on test set
    preds = clf.predict_batch(test_docs)
    metrics = evaluate(test_labels, preds)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
