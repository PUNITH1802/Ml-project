"""
model.py
--------
Training, evaluation, saving, and loading logic for the Spam Email Classifier.
Uses Naive Bayes (MultinomialNB) with CountVectorizer or TF-IDF vectorization.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from preprocess import preprocess_series

# Default paths for saved model artifacts
MODEL_PATH = "saved_model/model.pkl"
VECTORIZER_PATH = "saved_model/vectorizer.pkl"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_sample_dataset() -> pd.DataFrame:
    """
    Return a built-in sample dataset for quick demonstration.
    Each row has 'message' (raw text) and 'label' ('spam' or 'ham').
    """
    samples = [
        # SPAM examples
        ("WINNER!! You have been selected as the lucky winner of $1,000,000! Call now!", "spam"),
        ("Congratulations! Your mobile number has won £2000 in our prize draw. Text WIN to 87121", "spam"),
        ("FREE entry in 2 a weekly competition to win FA Cup Final tkts. Text FA to 87121 to receive entry", "spam"),
        ("URGENT! You have won a 1 week FREE membership in our award-winning adult web site!", "spam"),
        ("Naughty pics of celebs click here now http://freepics.com/celebs", "spam"),
        ("IMPORTANT! Please call 0800-333-444 to claim your prize before it expires!", "spam"),
        ("You are selected for a special promotion! Limited offer! Click now!", "spam"),
        ("Earn $5000 per week working from home. Guaranteed income. Apply now!", "spam"),
        ("Your account has been compromised. Verify your details at http://secure-login.com", "spam"),
        ("Buy Viagra, Cialis, and more! 80% off. No prescription needed. Order today.", "spam"),
        ("Get cheap loans now! Bad credit OK. Apply online. Instant approval.", "spam"),
        ("Win a brand new iPhone 15! Just complete our survey. Limited time!", "spam"),
        ("Congratulations! You've been chosen. Call 0800-CASH-NOW to claim your reward.", "spam"),
        ("Make money fast! Passive income guaranteed. Join thousands of winners.", "spam"),
        ("You owe taxes! Pay immediately or face arrest. Call IRS helpline now.", "spam"),
        ("Your parcel is ready. Confirm delivery by clicking: http://track.parcel-free.com", "spam"),
        ("HOT single women in your area are waiting to chat. Click here now!", "spam"),
        ("Special deal for you alone! 70% off luxury watches. Ships worldwide.", "spam"),
        ("FINAL NOTICE: Your subscription expires. Renew now to avoid being charged.", "spam"),
        ("Claim your $500 gift card. Survey takes 2 minutes. No credit card needed.", "spam"),
        ("Lose 30 lbs in 30 days! Doctor-approved fat burning secret. Order now.", "spam"),
        ("Investment opportunity! 300% returns guaranteed. Join our exclusive club.", "spam"),
        ("Your lottery ticket has been selected. Contact agent Smith at 555-0100.", "spam"),
        ("You are pre-approved for a $10,000 loan. No collateral needed. Apply today.", "spam"),
        ("Free ringtones! Download unlimited ringtones at no cost. Text RING to 99999.", "spam"),

        # HAM examples
        ("Hey, are you free this weekend? Thinking of having a small get-together.", "ham"),
        ("I'll pick you up at 7pm for dinner. Let me know if that works for you.", "ham"),
        ("Can you send me the project files? I need them before the meeting tomorrow.", "ham"),
        ("Happy birthday! Hope you have a wonderful day with family and friends.", "ham"),
        ("Let's schedule a call for Monday at 10am to discuss the quarterly report.", "ham"),
        ("Thanks for dinner last night, it was really nice catching up with you!", "ham"),
        ("I'll be late to the office today due to traffic. Starting around 9:30.", "ham"),
        ("Can you review my pull request? I've fixed the bug we discussed yesterday.", "ham"),
        ("The meeting has been moved to 3pm in Conference Room B. Please update your calendar.", "ham"),
        ("Hope you're feeling better! Let me know if you need anything.", "ham"),
        ("I finished the report. Can you check it over before I send it to the client?", "ham"),
        ("Don't forget we have a team lunch today at noon in the cafeteria.", "ham"),
        ("Your package was delivered and left at the front door as requested.", "ham"),
        ("I've attached the invoice for last month. Please process at your earliest convenience.", "ham"),
        ("Are you joining the book club meeting this Thursday? We're discussing chapter 5.", "ham"),
        ("Quick question: which version of Python are we using for the new project?", "ham"),
        ("I submitted the expense report. Should receive reimbursement within 5 business days.", "ham"),
        ("Running a bit late for lunch. Order without me if the food arrives — be there in 10.", "ham"),
        ("Would you mind reviewing the presentation before I present it to the board?", "ham"),
        ("We're out of milk and eggs. Can you grab some on your way home?", "ham"),
        ("Thanks for the quick turnaround on those design files. They look great!", "ham"),
        ("I'll be working from home tomorrow. You can reach me on Slack anytime.", "ham"),
        ("The client approved the proposal! Great work everyone. Let's celebrate Friday.", "ham"),
        ("Could you send me the link to the shared drive where the documents are stored?", "ham"),
        ("My flight lands at 6pm. See you at the hotel lobby around 7?", "ham"),
        ("Let me know when you're available this week — would love to catch up.", "ham"),
        ("Your appointment is confirmed for Tuesday, April 12th at 2:00 PM.", "ham"),
        ("I've sent the contract over for signature. Please review section 3 carefully.", "ham"),
        ("Just checking in — how's the project coming along? Need any support?", "ham"),
        ("The server maintenance is scheduled for Sunday 2–4 AM. No action needed from you.", "ham"),
    ]

    df = pd.DataFrame(samples, columns=["message", "label"])
    return df


def prepare_data(df: pd.DataFrame, use_stemming: bool = False) -> pd.DataFrame:
    """
    Validate and preprocess a DataFrame for training.

    Args:
        df: DataFrame with 'message' and 'label' columns.
        use_stemming: Whether to apply stemming during preprocessing.

    Returns:
        DataFrame with an added 'cleaned' column.

    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {"message", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {list(df.columns)}")

    df = df.copy()
    df.dropna(subset=["message", "label"], inplace=True)
    df["label"] = df["label"].str.strip().str.lower()
    df["cleaned"] = preprocess_series(df["message"], use_stemming=use_stemming)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    vectorizer_type: str = "tfidf",
    test_size: float = 0.2,
    random_state: int = 42,
    use_stemming: bool = False,
) -> dict:
    """
    Train a Naive Bayes classifier on the provided dataset.

    Args:
        df: DataFrame with 'message' and 'label' columns.
        vectorizer_type: 'tfidf' or 'count' — selects the feature extractor.
        test_size: Fraction of data to use for testing (default 0.2).
        random_state: Seed for reproducibility.
        use_stemming: Whether to apply stemming to text during preprocessing.

    Returns:
        A dict containing:
            - 'model': Trained MultinomialNB classifier
            - 'vectorizer': Fitted vectorizer
            - 'metrics': Dict with accuracy, precision, recall, f1, confusion_matrix, report
            - 'X_test_raw': Test messages (for sample predictions display)
            - 'y_test': True test labels
            - 'y_pred': Predicted test labels
    """
    df = prepare_data(df, use_stemming=use_stemming)

    X = df["cleaned"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Store raw test messages (before cleaning) for display
    X_test_raw = df.loc[X_test.index, "message"].values

    # Select vectorizer
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    else:
        vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))

    # Fit and transform
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Naive Bayes
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["ham", "spam"]),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "vectorizer_type": vectorizer_type,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_distribution": y.value_counts().to_dict(),
    }

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
        "X_test_raw": X_test_raw,
        "y_test": y_test.values,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(text: str, model, vectorizer) -> dict:
    """
    Predict whether a single text message is spam or ham.

    Args:
        text: Raw text input from the user.
        model: Trained MultinomialNB classifier.
        vectorizer: Fitted vectorizer (CountVectorizer or TfidfVectorizer).

    Returns:
        A dict with:
            - 'label': 'spam' or 'ham'
            - 'confidence': Float between 0 and 1
            - 'spam_prob': Probability of being spam
            - 'ham_prob': Probability of being ham
    """
    from preprocess import clean_text

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]

    # class order is alphabetical: ham=0, spam=1
    classes = model.classes_
    class_proba = dict(zip(classes, proba))

    spam_prob = class_proba.get("spam", 0.0)
    ham_prob = class_proba.get("ham", 0.0)
    label = "spam" if spam_prob > ham_prob else "ham"
    confidence = max(spam_prob, ham_prob)

    return {
        "label": label,
        "confidence": float(confidence),
        "spam_prob": float(spam_prob),
        "ham_prob": float(ham_prob),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, vectorizer, model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH):
    """
    Save the trained model and vectorizer to disk using pickle.

    Args:
        model: Trained MultinomialNB classifier.
        vectorizer: Fitted vectorizer.
        model_path: File path to save the model.
        vectorizer_path: File path to save the vectorizer.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_model(model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH):
    """
    Load a previously saved model and vectorizer from disk.

    Args:
        model_path: File path to the saved model.
        vectorizer_path: File path to the saved vectorizer.

    Returns:
        Tuple of (model, vectorizer), or (None, None) if files don't exist.
    """
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def model_exists(model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH) -> bool:
    """Check whether a saved model exists on disk."""
    return os.path.exists(model_path) and os.path.exists(vectorizer_path)
