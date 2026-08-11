"""
Unified Streamlit app: price visualization + sentiment analytics + RAG chatbot.
Run with:  streamlit run app.py
(from the same directory that contains data/ and models/, produced by the notebook.)
"""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy import stats
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

st.set_page_config(page_title="Sentiment & RAG Dashboard", layout="wide")
st.title("Financial News Sentiment & RAG Dashboard")

@st.cache_data
def load_data():
    news = pd.read_csv(DATA_DIR / "news_with_sentiment.csv")
    daily = pd.read_csv(DATA_DIR / "prices_daily.csv")
    return news, daily

@st.cache_resource
def load_rag():
    index = faiss.read_index(str(MODELS_DIR / "news_index.faiss"))
    metadata = pd.read_csv(MODELS_DIR / "news_index_metadata.csv")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return index, metadata, embed_model

news, daily = load_data()
tickers = sorted(news["ticker"].unique())

tab1, tab2, tab3 = st.tabs(["Price & Sentiment", "Statistical Test", "Ask (RAG)"])

with tab1:
    ticker = st.selectbox("Ticker", tickers)
    price_t = daily[daily["ticker"] == ticker].sort_values("date")
    news_t = news[news["ticker"] == ticker]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{ticker} - Price")
        st.line_chart(price_t.set_index("date")["close"])
    with col2:
        st.subheader(f"{ticker} - Sentiment Distribution")
        st.bar_chart(news_t["sentiment_label"].value_counts())

    st.subheader("Recent Headlines")
    st.dataframe(
        news_t[["published_at", "title", "sentiment_label", "sentiment_confidence"]]
        .sort_values("published_at", ascending=False).head(20),
        use_container_width=True,
    )

with tab2:
    st.subheader("Welch's t-test: sentiment vs next-day direction")
    if (DATA_DIR / "train.csv").exists():
        train = pd.read_csv(DATA_DIR / "train.csv")
        up = train[train["label_up"] == 1]["prob_positive"]
        down = train[train["label_up"] == 0]["prob_positive"]
        t_stat, p_val = stats.ttest_ind(up, down, equal_var=False)
        st.metric("t-statistic", f"{t_stat:.3f}")
        st.metric("p-value", f"{p_val:.4f}")
        st.write("Significant (p<0.05)" if p_val < 0.05 else "Not significant (p>=0.05)")
    else:
        st.info("Run the notebook's Section 6-7 first to generate train.csv.")

    if (DATA_DIR / "hourly_test_results.csv").exists():
        st.subheader("Intraday horizons")
        st.dataframe(pd.read_csv(DATA_DIR / "hourly_test_results.csv"), use_container_width=True)

with tab3:
    st.subheader("Ask why a stock moved")
    query = st.text_input("Question", placeholder="Why is NVDA sentiment negative this week?")
    rag_ticker = st.selectbox("Filter to ticker (optional)", ["All"] + tickers)
    days_back = st.slider("Only articles from the last N days", 1, 30, 7)

    if st.button("Ask") and query:
        index, metadata, embed_model = load_rag()
        query_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, 100)
        results = metadata.iloc[indices[0]].copy()
        results["similarity"] = scores[0]
        if rag_ticker != "All":
            results = results[results["ticker"] == rag_ticker]
        results["published_at"] = pd.to_datetime(results["published_at"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        results = results[results["published_at"] >= cutoff]
        results = results.sort_values("similarity", ascending=False).head(8)

        if results.empty:
            st.warning("No matching articles found - try widening the day range or removing the ticker filter.")
        else:
            st.write("**Retrieved headlines:**")
            for _, row in results.iterrows():
                st.write(f"- [{row['ticker']}] ({row['similarity']:.3f}) {row['title']} — *{row['sentiment_label']}*")

            context = "\n".join(
                f"- \"{row['title']}\" (FinBERT sentiment: {row['sentiment_label']})"
                for _, row in results.iterrows()
            )
            prompt = f"""You are a financial news analyst. Answer using ONLY the headlines below. Be concise
(3-5 sentences) and synthesize a takeaway. If the headlines don't contain enough information, say so.

Headlines:
{context}

Question: {query}

Answer:"""
            try:
                resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=120)
                resp.raise_for_status()
                st.write("**Answer:**")
                st.write(resp.json().get("response", "").strip())
            except requests.exceptions.ConnectionError:
                st.error("Could not reach Ollama. Make sure it's running (`ollama serve`).")
