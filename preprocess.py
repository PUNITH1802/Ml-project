"""
preprocess.py
-------------
Text cleaning and preprocessing utilities for the Spam Email Classifier.
Handles lowercasing, punctuation removal, stopword removal, and tokenization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data on first import
def download_nltk_data():
    """Download required NLTK datasets if not already present."""
    required = [("corpora/stopwords", "stopwords"), ("tokenizers/punkt", "punkt")]
    for path, name in required:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_data()

# Initialize stemmer and stopword list
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str, use_stemming: bool = False) -> str:
    """
    Clean and preprocess a single text string.

    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation and digits
    5. Remove extra whitespace
    6. Remove stopwords
    7. Optionally apply stemming

    Args:
        text: Raw input text (email body or message).
        use_stemming: If True, apply Porter stemming to each word.

    Returns:
        Cleaned and preprocessed text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # 4. Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # 5. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in _stop_words and len(word) > 1]

    # 7. Optional stemming
    if use_stemming:
        tokens = [_stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


def preprocess_series(series, use_stemming: bool = False):
    """
    Apply clean_text to every element in a pandas Series.

    Args:
        series: A pandas Series of raw text strings.
        use_stemming: If True, apply stemming to each token.

    Returns:
        A pandas Series of cleaned text strings.
    """
    return series.apply(lambda text: clean_text(text, use_stemming=use_stemming))
