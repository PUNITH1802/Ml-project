# Spam Email Classifier

A beginner-friendly, production-style Machine Learning project that detects spam emails and messages using Natural Language Processing (NLP) and Naive Bayes classification.

## Features

- **Text Preprocessing** — lowercase, URL removal, punctuation stripping, stopword removal, optional stemming
- **Feature Extraction** — switch between TF-IDF and Count Vectorizer
- **Naive Bayes Classifier** — MultinomialNB trained on labeled email data
- **Evaluation Metrics** — accuracy, precision, recall, F1-score, confusion matrix, classification report
- **Interactive UI** — Streamlit web app with live spam detection and confidence scores
- **CSV Upload** — bring your own labeled dataset (`message`, `label` columns)
- **Model Persistence** — save/load trained model using pickle

## Project Structure

```
spam-classifier/
├── app.py           # Streamlit UI (main entry point)
├── model.py         # Training, prediction, save/load logic
├── preprocess.py    # Text cleaning and preprocessing utilities
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK data (auto-handled on first run)

The app downloads `stopwords` and `punkt` automatically via NLTK on first launch.

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501) by default.

## How It Works

### 1. Preprocessing (`preprocess.py`)

Raw text is cleaned through a pipeline:
1. Lowercase conversion
2. URL removal (regex)
3. HTML tag stripping
4. Punctuation and digit removal
5. Stopword removal (NLTK English stopwords)
6. Optional: Porter Stemming

### 2. Feature Extraction

| Method | Description |
|--------|-------------|
| **TF-IDF** | Down-weights frequent terms, highlights distinctive words |
| **Count Vectorizer** | Raw term frequency counts |

Both use unigrams + bigrams (`ngram_range=(1, 2)`) and cap at 5,000 features.

### 3. Model — Multinomial Naive Bayes

- Probabilistic classifier well-suited for text
- Trained using `scikit-learn`'s `MultinomialNB(alpha=0.1)`
- Outputs both a label (`spam`/`ham`) and class probabilities

### 4. Evaluation

- **Accuracy** — overall correct predictions
- **Precision** — of all messages predicted spam, how many were actually spam
- **Recall** — of all actual spam messages, how many were caught
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — visualizes TP, TN, FP, FN

## Using Your Own Dataset

Upload a CSV with these columns via the sidebar:

| Column | Description |
|--------|-------------|
| `message` | Raw email or message text |
| `label` | `spam` or `ham` |

## Model Persistence

| Action | Description |
|--------|-------------|
| **Save Model** | Saves `model.pkl` and `vectorizer.pkl` to `saved_model/` |
| **Load Model** | Loads a previously saved model from disk |

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| pandas / NumPy | Data manipulation |
| scikit-learn | ML pipeline |
| NLTK | NLP preprocessing |
| Streamlit | Web UI |
| Matplotlib / Seaborn | Visualizations |
| pickle | Model serialization |

## API / Module Reference

### `preprocess.py`
- `clean_text(text, use_stemming)` — Clean a single text string
- `preprocess_series(series, use_stemming)` — Apply cleaning to a pandas Series

### `model.py`
- `load_sample_dataset()` — Returns built-in labeled DataFrame
- `prepare_data(df, use_stemming)` — Validate and preprocess a DataFrame
- `train_model(df, vectorizer_type, test_size, ...)` — Train and evaluate classifier
- `predict(text, model, vectorizer)` — Predict spam/ham for a single message
- `save_model(model, vectorizer)` — Persist model to disk with pickle
- `load_model()` — Load model from disk
- `model_exists()` — Check if a saved model is available

## License

MIT
