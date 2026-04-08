"""
app.py
------
Streamlit UI for the Spam Email Classifier.

Run with:
    streamlit run app.py

Features:
- Train a Naive Bayes classifier on a built-in or user-uploaded dataset
- Switch between TF-IDF and CountVectorizer
- Type any email/message and get a spam prediction with confidence score
- View evaluation metrics: accuracy, confusion matrix, classification report
- See sample predictions on test data
- Save / reload the trained model with pickle
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# Import project modules (must be in same directory)
from model import (
    load_sample_dataset,
    train_model,
    predict,
    save_model,
    load_model,
    model_exists,
    MODEL_PATH,
    VECTORIZER_PATH,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .spam-badge {
        background-color: #FF4B4B;
        color: white;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .ham-badge {
        background-color: #21C55D;
        color: white;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — configuration panel
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

vectorizer_type = st.sidebar.radio(
    "Feature Extractor",
    options=["tfidf", "count"],
    format_func=lambda x: "TF-IDF Vectorizer" if x == "tfidf" else "Count Vectorizer",
    help="TF-IDF down-weights common words; CountVectorizer uses raw term frequencies.",
)

use_stemming = st.sidebar.checkbox(
    "Apply Stemming",
    value=False,
    help="Reduce words to their root form using Porter Stemmer.",
)

test_split = st.sidebar.slider(
    "Test Split Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
    help="Fraction of the dataset used for evaluation.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Upload Your Dataset (optional)")
uploaded_file = st.sidebar.file_uploader(
    "CSV with 'message' and 'label' columns",
    type=["csv"],
    help="Column 'label' must contain 'spam' or 'ham'.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Model Persistence")
col_save, col_load = st.sidebar.columns(2)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "train_results" not in st.session_state:
    st.session_state.train_results = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# ---------------------------------------------------------------------------
# Load dataset (uploaded or built-in sample)
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        if "message" not in df_raw.columns or "label" not in df_raw.columns:
            st.sidebar.error("CSV must have 'message' and 'label' columns.")
            df_raw = load_sample_dataset()
            st.sidebar.info("Falling back to built-in sample dataset.")
        else:
            st.sidebar.success(f"Loaded {len(df_raw)} rows from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        df_raw = load_sample_dataset()
else:
    df_raw = load_sample_dataset()

st.session_state.dataset = df_raw

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
st.title("🛡️ Spam Email Classifier")
st.markdown(
    "An end-to-end Machine Learning project using **Naive Bayes** + **NLP** to detect spam emails and messages."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_classify, tab_train, tab_metrics, tab_samples, tab_data = st.tabs([
    "🔍 Classify Message",
    "🏋️ Train Model",
    "📊 Evaluation Metrics",
    "📋 Sample Predictions",
    "🗄️ Dataset",
])

# ---------------------------------------------------------------------------
# TAB 1 — Classify Message
# ---------------------------------------------------------------------------
with tab_classify:
    st.header("Classify a Message or Email")
    st.markdown("Type or paste any email body or message below and click **Check Spam**.")

    user_input = st.text_area(
        "Enter your message:",
        height=180,
        placeholder="e.g., Congratulations! You've won a $1,000,000 prize! Click here to claim it now!",
    )

    if st.button("🔍 Check Spam", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a message before checking.")
        elif st.session_state.model is None:
            st.warning("No trained model found. Please go to the **Train Model** tab first.")
        else:
            result = predict(user_input, st.session_state.model, st.session_state.vectorizer)
            label = result["label"].upper()
            confidence = result["confidence"] * 100
            spam_pct = result["spam_prob"] * 100
            ham_pct = result["ham_prob"] * 100

            st.markdown("---")
            col_r1, col_r2 = st.columns([1, 2])

            with col_r1:
                st.markdown("#### Prediction")
                if result["label"] == "spam":
                    st.markdown('<span class="spam-badge">🚨 SPAM</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="ham-badge">✅ NOT SPAM</span>', unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{confidence:.1f}%`")

            with col_r2:
                st.markdown("#### Probability Breakdown")
                prob_df = pd.DataFrame({
                    "Category": ["Ham (Legitimate)", "Spam"],
                    "Probability": [ham_pct, spam_pct],
                })
                fig, ax = plt.subplots(figsize=(5, 2))
                colors = ["#21C55D" if p == max(spam_pct, ham_pct) and c == "Spam" else
                          "#21C55D" if p == max(spam_pct, ham_pct) and c == "Ham (Legitimate)" else
                          "#E5E7EB"
                          for p, c in zip(prob_df["Probability"], prob_df["Category"])]
                # Simpler coloring: red for spam bar, green for ham bar
                bar_colors = ["#21C55D", "#FF4B4B"]
                bars = ax.barh(prob_df["Category"], prob_df["Probability"], color=bar_colors, height=0.5)
                ax.set_xlim(0, 100)
                ax.set_xlabel("Probability (%)")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                for bar, val in zip(bars, prob_df["Probability"]):
                    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # Try to auto-load saved model on first visit
    if st.session_state.model is None and model_exists():
        m, v = load_model()
        if m is not None:
            st.session_state.model = m
            st.session_state.vectorizer = v
            st.info("Loaded previously saved model from disk.")

# ---------------------------------------------------------------------------
# TAB 2 — Train Model
# ---------------------------------------------------------------------------
with tab_train:
    st.header("Train the Classifier")
    st.markdown(
        f"Dataset: **{len(df_raw)} messages** | "
        f"Vectorizer: **{'TF-IDF' if vectorizer_type == 'tfidf' else 'CountVectorizer'}** | "
        f"Stemming: **{'On' if use_stemming else 'Off'}** | "
        f"Test Split: **{int(test_split * 100)}%**"
    )

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training Naive Bayes classifier..."):
            try:
                results = train_model(
                    df_raw,
                    vectorizer_type=vectorizer_type,
                    test_size=test_split,
                    use_stemming=use_stemming,
                )
                st.session_state.model = results["model"]
                st.session_state.vectorizer = results["vectorizer"]
                st.session_state.train_results = results
                st.success("Model trained successfully! View results in the **Evaluation Metrics** and **Sample Predictions** tabs.")
            except ValueError as e:
                st.error(f"Training failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    if st.session_state.train_results is not None:
        metrics = st.session_state.train_results["metrics"]
        st.markdown("---")
        st.subheader("Quick Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall", f"{metrics['recall']:.2%}")
        c4.metric("F1-Score", f"{metrics['f1']:.2%}")

    # Model save / load buttons
    st.markdown("---")
    st.subheader("Model Persistence")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("💾 Save Model to Disk", use_container_width=True):
            if st.session_state.model is None:
                st.warning("Train a model first.")
            else:
                save_model(st.session_state.model, st.session_state.vectorizer)
                st.success(f"Model saved to `{MODEL_PATH}` and `{VECTORIZER_PATH}`")
    with sc2:
        if st.button("📂 Load Model from Disk", use_container_width=True):
            if model_exists():
                m, v = load_model()
                st.session_state.model = m
                st.session_state.vectorizer = v
                st.success("Model loaded successfully from disk.")
            else:
                st.warning("No saved model found. Train and save one first.")

# ---------------------------------------------------------------------------
# TAB 3 — Evaluation Metrics
# ---------------------------------------------------------------------------
with tab_metrics:
    st.header("Evaluation Metrics")

    if st.session_state.train_results is None:
        st.info("Train a model first to see evaluation metrics.")
    else:
        results = st.session_state.train_results
        metrics = results["metrics"]

        # Summary metrics
        st.subheader("Performance Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("Precision (Spam)", f"{metrics['precision']:.4f}")
        m3.metric("Recall (Spam)", f"{metrics['recall']:.4f}")
        m4.metric("F1-Score (Spam)", f"{metrics['f1']:.4f}")

        st.markdown("---")
        col_cm, col_cr = st.columns(2)

        # Confusion Matrix
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = metrics["confusion_matrix"]
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Ham (Predicted)", "Spam (Predicted)"],
                yticklabels=["Ham (Actual)", "Spam (Actual)"],
                ax=ax_cm,
                linewidths=0.5,
                linecolor="gray",
            )
            ax_cm.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

        # Classification Report
        with col_cr:
            st.subheader("Classification Report")
            st.code(metrics["report"], language="text")

        st.markdown("---")
        st.subheader("Dataset Info")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"- **Train samples:** {metrics['train_size']}")
            st.markdown(f"- **Test samples:** {metrics['test_size']}")
            st.markdown(f"- **Vectorizer:** `{metrics['vectorizer_type'].upper()}`")
        with info_col2:
            dist = metrics["class_distribution"]
            st.markdown(f"- **Ham messages:** {dist.get('ham', 0)}")
            st.markdown(f"- **Spam messages:** {dist.get('spam', 0)}")
            total = sum(dist.values())
            st.markdown(f"- **Total messages:** {total}")

        # Class distribution bar chart
        st.subheader("Class Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(4, 2.5))
        labels_d = list(dist.keys())
        values_d = [dist[k] for k in labels_d]
        colors_d = ["#21C55D" if l == "ham" else "#FF4B4B" for l in labels_d]
        ax_dist.bar(labels_d, values_d, color=colors_d, width=0.4)
        for i, v in enumerate(values_d):
            ax_dist.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
        ax_dist.set_ylabel("Count")
        ax_dist.spines["top"].set_visible(False)
        ax_dist.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close()

# ---------------------------------------------------------------------------
# TAB 4 — Sample Predictions
# ---------------------------------------------------------------------------
with tab_samples:
    st.header("Sample Predictions on Test Set")

    if st.session_state.train_results is None:
        st.info("Train a model first to see sample predictions.")
    else:
        results = st.session_state.train_results
        X_test_raw = results["X_test_raw"]
        y_test = results["y_test"]
        y_pred = results["y_pred"]

        # Build predictions table
        sample_data = []
        for msg, actual, pred in zip(X_test_raw, y_test, y_pred):
            result_row = predict(msg, st.session_state.model, st.session_state.vectorizer)
            sample_data.append({
                "Message (truncated)": msg[:80] + ("..." if len(msg) > 80 else ""),
                "Actual": actual.upper(),
                "Predicted": pred.upper(),
                "Spam Prob": f"{result_row['spam_prob']:.2%}",
                "Correct": "✅" if actual == pred else "❌",
            })

        sample_df = pd.DataFrame(sample_data)

        # Color rows
        def highlight_row(row):
            if row["Correct"] == "❌":
                return ["background-color: #ffe0e0"] * len(row)
            return [""] * len(row)

        st.dataframe(
            sample_df.style.apply(highlight_row, axis=1),
            use_container_width=True,
            height=400,
        )
        st.caption(f"Showing all {len(sample_df)} test samples. Red rows indicate misclassifications.")

# ---------------------------------------------------------------------------
# TAB 5 — Dataset View
# ---------------------------------------------------------------------------
with tab_data:
    st.header("Dataset Preview")

    source = "Uploaded CSV" if uploaded_file is not None else "Built-in Sample Dataset"
    st.markdown(f"**Source:** {source} | **Total rows:** {len(df_raw)}")

    dist_vals = df_raw["label"].str.lower().value_counts()
    dc1, dc2 = st.columns(2)
    dc1.metric("Ham messages", dist_vals.get("ham", 0))
    dc2.metric("Spam messages", dist_vals.get("spam", 0))

    st.dataframe(df_raw, use_container_width=True, height=400)

    # Download dataset as CSV
    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Dataset as CSV",
        data=csv_bytes,
        file_name="spam_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<small>Spam Email Classifier · Built with Python, Scikit-learn, NLTK & Streamlit · "
    "Model: Multinomial Naive Bayes · Features: TF-IDF / CountVectorizer</small>",
    unsafe_allow_html=True,
)
