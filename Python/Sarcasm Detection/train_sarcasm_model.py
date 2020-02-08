#!/usr/bin/env python3
# News Headlines Dataset For Sarcasm Detection
import glob
import ipykernel
import os
import time
import nltk
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from string import punctuation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Nadam
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from tensorflow_core.python.keras.layers.recurrent_v2 import cudnn_lstm
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
tf.config.experimental.list_physical_devices('GPU')
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')

st = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def rep_sent_vector(str1=''):
    str1 = str1.lower()
    # tags = st.tag(str1.split())
    # for tag in tags:
    #     # print(tag)
    #     if tag[1] != 'O':
    #         str1 = str1.replace(tag[0], '')
    words = nltk.tokenize.word_tokenize(str1)  # [words for words in str1.split() if words not in string.punctuation]
    # words = [w for w in words if w not in stopwords.words('english')]
    # lemmatizer = WordNetLemmatizer()
    # words = [lemmatizer.lemmatize(w) for w in words if
    #          ((w not in punctuation) or (w not in stopwords.words('english')))]
    if len(words) == 0:
        return [[0]*100]*11
    else:
        # print(words)
        # print(np.array(Word2Vec([words], min_count=1, workers=10, sorted_vocab=0).wv.vectors).shape)
        return Word2Vec([words], min_count=1, workers=8, sorted_vocab=0,iter=100).wv.vectors


def buildTrainData():
    # x_train = []
    df = pd.DataFrame(pd.concat(map(lambda x: pd.read_json(x, lines=True), glob.glob(os.path.join('', "*.json")))))
    x_train = df['headline'].apply(lambda x: rep_sent_vector(x)).tolist()

    # x_train = pad_sequences(x_train, maxlen=200, padding='pre', truncating='pre', value=0)
    x_train = pad_sequences(x_train, padding='pre', value=0, maxlen=200)
    np.save('x_train', np.array(x_train,dtype=np.float))
    print("Process Completed Successfully")
    # y_train = to_categorical(df['is_sarcastic'].tolist())
    # print(len(x_train))  # 55328
    # x_test = np.array(x_train[50000:])
    # y_test = np.array(y_train[50000:])
    # x_train = np.array(x_train[0:50000])
    # y_train = np.array(y_train[0:50000])
    # print(x_train.shape, y_train.shape)
    # return x_train, y_train, x_test, y_test


def trainBiLstmModel(x_train, y_train, x_test, y_test):
    model = Sequential()
    data_dim = 100
    time_steps = 200
    model.add(LSTM(128, return_sequences=True, input_shape=(time_steps, data_dim)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    # model.add(cudnn_lstm())
    model.add(Dense(2, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=1000, epochs=200, validation_data=(x_test, y_test), workers=8,
              use_multiprocessing=True)
    model.save('./model/sarcasm_model.h5')


# rep_sent_vector("former versace store clerk sues over secret 'black code' for minority shoppers")
def process():
    print(time.asctime(time.localtime(time.time())))
    try:
        x_train, y_train, x_test, y_test = buildTrainData()
        # trainBiLstmModel(x_train, y_train, x_test, y_test)
        # print(time.asctime(time.localtime(time.time())))
    except BaseException as e:
        print(time.asctime(time.localtime(time.time())))
        print(e)


process()
# print(rep_sent_vector(''))
# rep_sent_vector("j.k. rowling wishes snape happy birthday in the most magical way")
