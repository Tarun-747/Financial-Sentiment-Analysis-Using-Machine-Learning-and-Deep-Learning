from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_lstm_model(vocab_size=5000, max_len=100, embed_dim=64):
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len),
        LSTM(64, dropout=0.4, recurrent_dropout=0.4),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

