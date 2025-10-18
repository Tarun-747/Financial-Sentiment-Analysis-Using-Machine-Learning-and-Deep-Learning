import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.preprocessing import preprocess_text
from src.features import get_tfidf_features, get_padded_sequences
from src.train_ml import train_logistic_regression, evaluate_ml
from src.train_dl import build_lstm_model
from src.evaluate import plot_training_history
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model

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

# Encode labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['sentiment'])

# Split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF + Logistic Regression
X_train_tfidf, vectorizer = get_tfidf_features(X_train)
X_test_tfidf = vectorizer.transform(X_test)
lr_model = train_logistic_regression(X_train_tfidf, y_train)
evaluate_ml(lr_model, X_test_tfidf, y_test, encoder.classes_)

# LSTM Model
X_train_pad, tokenizer = get_padded_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

model = build_lstm_model()
history = model.fit(X_train_pad, y_train, validation_split=0.4, epochs=15, batch_size=64)
plot_training_history(history)
model.save("models/lstm_model.h5")
