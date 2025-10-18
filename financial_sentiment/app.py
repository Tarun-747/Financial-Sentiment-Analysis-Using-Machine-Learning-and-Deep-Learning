import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import preprocess_text

# Streamlit setup
st.set_page_config(page_title="Financial Sentiment Dashboard", layout="wide")
st.title("💹 Financial Sentiment Analysis Dashboard")

@st.cache_resource
def load_models():
    lr_model = joblib.load("models/logistic_reg.pkl")
    lstm_model = load_model("models/lstm_model.h5")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    tokenizer = joblib.load("models/tokenizer.pkl")
    return lr_model, lstm_model, vectorizer, tokenizer

lr_model, lstm_model, vectorizer, tokenizer = load_models()

st.sidebar.header("Choose Model")
model_choice = st.sidebar.radio("Model", ["Logistic Regression", "LSTM (Deep Learning)"])

st.sidebar.header("Input Type")
input_choice = st.sidebar.radio("Choose Input", ["Single Text", "Upload CSV"])

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# --- SINGLE TEXT ---
if input_choice == "Single Text":
    text = st.text_area("Enter financial text:", 
                        "The company's revenue grew by 15% this quarter despite inflation pressures.")
    
    if st.button("Analyze Sentiment"):
        cleaned = preprocess_text(text)
        if model_choice == "Logistic Regression":
            X = vectorizer.transform([cleaned])
            pred = lr_model.predict(X)[0]
        else:
            seq = tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=100)
            pred = np.argmax(lstm_model.predict(pad), axis=1)[0]

        st.success(f"Predicted Sentiment: **{sentiment_map[pred]}**")

# --- CSV UPLOAD ---
else:
    uploaded = st.file_uploader("Upload a CSV with a column 'sentence'", type=["csv"])
    if uploaded:
        # Load dataset
        df = pd.read_csv(
            "data/FinancialPhraseBank.csv",
            sep=None,           # auto-detect separator
            engine='python',    # needed for sep=None
            header=None,
            encoding='latin1'
        )

        df.columns = ['sentiment', 'sentence']
        # Preprocess
        df['clean_text'] = df['sentence'].apply(preprocess_text)
        # df['clean_text'] = df['sentence'].apply(preprocess_text)
        
        if model_choice == "Logistic Regression":
            X = vectorizer.transform(df['clean_text'])
            df['predicted'] = lr_model.predict(X)
        else:
            seq = tokenizer.texts_to_sequences(df['clean_text'])
            pad = pad_sequences(seq, maxlen=100)
            df['predicted'] = np.argmax(lstm_model.predict(pad), axis=1)

        df['predicted'] = df['predicted'].map(sentiment_map)
        st.dataframe(df[['sentence', 'predicted']].head())

        st.subheader("📊 Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='predicted', data=df, ax=ax)
        plt.title("Predicted Sentiment Distribution")
        st.pyplot(fig)

        st.download_button("Download Results CSV", df.to_csv(index=False).encode('utf-8'), "sentiment_results.csv", "text/csv")
