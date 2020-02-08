'''https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection'''
import glob
import os
import time
from string import punctuation

import nltk
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# tf.debugging.set_log_device_placement(True)
# print(device_lib.list_local_devices())
# tf.config.experimental.list_physical_devices('GPU')
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')

st = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
punctuation = ["'", ",", '"', ";"':']


def rep_sent_vector(str1=''):
 str1 = str1.lower()
 # for p in punctuation:
 # str1.replace(p, "")
 # tags = st.tag(str1.split())
 # for tag in tags:
 # # print(tag)
 # if tag[1] != 'O':
 # str1 = str1.replace(tag[0], '')
 words = nltk.tokenize.word_tokenize(str1) # [words for words in str1.split() if words not in string.punctuation]
 # words = [w for w in words if w not in stopwords.words('english')]
 # lemmatizer = WordNetLemmatizer()
 # words = [lemmatizer.lemmatize(w) for w in words if
 # ((w not in punctuation) and (w not in stopwords.words('english')))]
 return words


def buildTrainData():
 df = pd.DataFrame(pd.concat(map(lambda x: pd.read_json(x, lines=True), glob.glob(os.path.join('', "*.json")))))
 lemwords = df['headline'].apply(lambda x: (rep_sent_vector(x))).tolist()
 maxWords = len(max(lemwords))
 tokenizer = Tokenizer(num_words=20000)
 tokenizer.fit_on_texts(lemwords)
 sequences = tokenizer.texts_to_sequences(lemwords)
 x_train = pad_sequences(sequences, maxlen=maxWords)
 y_train = df['is_sarcastic'].tolist()
 x_test = np.array(x_train[50000:], dtype=np.float)
 y_test = np.array(y_train[50000:], dtype=np.float)
 x_train = np.array(x_train[0:50000], dtype=np.float)
 y_train = np.array(y_train[0:50000], dtype=np.float)
 print(x_train.shape, y_train.shape)
 return x_train, y_train, x_test, y_test, maxWords


def trainBiLstmModel(x_train, y_train, x_test, y_test, max_token):
 model = Sequential()
 model.add(Embedding(20000, 128, input_length=max_token))
 model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1, dropout=0.1, activation='relu')))
 model.add(Dense(1, activation='sigmoid'))
 model.compile(loss='binary_crossentropy',
 optimizer='adam',
 metrics=['accuracy'])
 model.fit(x_train, y_train, batch_size=500, epochs=17, validation_data=(x_test, y_test), workers=8,
 use_multiprocessing=True, verbose=2)
 model.save('./model/sarcasm_model.h5', overwrite=True)


def process():
 print(time.asctime(time.localtime(time.time())))
 try:
 x_train, y_train, x_test, y_test, max_token = buildTrainData()
 trainBiLstmModel(x_train, y_train, x_test, y_test, max_token)
 print(time.asctime(time.localtime(time.time())))
 except BaseException as e:
 print(time.asctime(time.localtime(time.time())))
 print(e)


process()