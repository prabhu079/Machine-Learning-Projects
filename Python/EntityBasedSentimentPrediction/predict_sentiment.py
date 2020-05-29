import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Embedding, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def process_sentence(str1):
    words = str1.split(" ")
    return words


def pre_Process_Data(df, status="train"):
    lemwords = (df["Entity"] + df['Sentence']).apply(lambda x: process_sentence(x)).tolist()
    maxWords = 64
    tokenizer = Tokenizer(num_words=128)
    tokenizer.fit_on_texts(lemwords)
    sequences = tokenizer.texts_to_sequences(lemwords)
    x_train = pad_sequences(sequences, maxlen=maxWords, dtype='float32')
    if status == "test":
        return x_train
    else:
        y_train = df['Sentiment'].replace("positive", 1).replace("negative", 0).tolist()
        x_test = np.array(x_train[3600:], dtype=np.float)
        y_test = np.array(y_train[3600:], dtype=np.float)
        x_train = np.array(x_train[0:3600], dtype=np.float)
        y_train = np.array(y_train[0:3600], dtype=np.float)
        return x_train, y_train, x_test, y_test, maxWords


def trainBiLstmModel(x_train, y_train, x_test, y_test, max_token):
    model = Sequential()
    model.add(Embedding(128, 8, input_length=max_token))
    model.add(LSTM(4, dropout=0.5, recurrent_dropout=0.5))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.008),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=512, epochs=64, validation_data=(x_test, y_test), workers=8,
              use_multiprocessing=True, verbose=2)
    model.save('./model/sentiment_model.h5', overwrite=True)


def process():
    df = pd.read_excel("./data/Entity_sentiment_trainV2.xlsx")
    x_train, y_train, x_val, y_val, max_words = pre_Process_Data(df)
    trainBiLstmModel(x_train, y_train, x_val, y_val, max_words)
    df_test = pd.read_excel("./data/Entity_sentiment_testV2.xlsx")
    x_test = pre_Process_Data(df_test, "test")
    model = load_model('./model/sentiment_model.h5')
    df_test["Sentiment"] = model.predict_classes(x_test)
    df_test["Sentiment"] = df_test["Sentiment"].replace(1, "positive").replace(0, "negative")
    df_test.to_excel("./data/Entity_sentiment_testV2.xlsx", columns=["Sentence", "Entity", "Sentiment"], index=False)


process()
