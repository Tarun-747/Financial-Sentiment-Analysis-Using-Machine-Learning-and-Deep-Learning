from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

def get_tfidf_features(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    return X, vectorizer

def get_padded_sequences(texts, max_words=5000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    seq = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seq, maxlen=max_len)
    joblib.dump(tokenizer, "models/tokenizer.pkl")
    return X, tokenizer
