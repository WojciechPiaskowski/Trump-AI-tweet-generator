# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Embedding,\
    Conv1D, MaxPooling1D, GlobalMaxPool1D, Conv2D, Flatten, GlobalMaxPool2D, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model, load_model
import os
import nltk
from gensim.models import Word2Vec, KeyedVectors
import re
# nltk.download('punkt')
import tensorflow.keras.callbacks as ES

# style conifg
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
sns.set_style('whitegrid')

# import Trump tweets (dataset from Kaggle.com, scrapped from twitter)
df = pd.read_csv('realdonaldtrump.csv', sep=',')
# shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# first look at the data
df.head()
df = df[['content', 'id']]

# remove at least some of the links/unwanted strings
df['content'] = df['content'].apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ',', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'pic.twitter', ',', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'.com', ',', x))

# size of embeding dimension (it is set to 300 as that is the number of dimensions in google word2vec network)
embd_size = 300

# call google news word2vec model
# you can find it here: https://code.google.com/archive/p/word2vec/
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

all_words = []

for i in df['content']:
        for j in i.split():
            all_words.append(j.lower())

all_words = all_words[:200000]

all_words_emb = np.empty((len(all_words), embd_size))
j = 0
for i in all_words:
    try:
        all_words_emb[j] = word2vec[i]
        j += 1
    except:
        continue

lag = 10
N = all_words_emb.shape[0]-lag

X = np.empty((N, lag, embd_size))
Y = np.empty((N, embd_size,))
j = 0
for i in range(N-1):
    x = all_words_emb[i:lag+i]
    y = all_words_emb[lag+i]
    X[j] = x
    Y[j] = y
    j += 1

X_train = X[:150000]
Y_train = Y[:150000]
X_test = X[150000:]
Y_test = Y[150000:]

i = Input(shape=(10, 300,))
x = LSTM(1024, return_sequences=True)(i)
x = Dropout(0.2)(x)
x = LSTM(1024)(x)
x = Dropout(0.2)(x)
x = Dense(embd_size)(x)

cb = ES.EarlyStopping(monitor='val_loss', patience=2)

model1 = Model(i, x)
model1.compile(optimizer='adam', loss='mse', metrics=['mse'])
r1 = model1.fit(X_train, Y_train, epochs=2, validation_data=(X_test, Y_test), batch_size=200, callbacks=[cb])

plt.plot(r1.history['loss'], label='loss')
plt.plot(r1.history['val_loss'], label=['val_loss'])
plt.legend()

idx = np.random.randint(X.shape[0])
input = X[idx]
input = input.reshape(1, 10, 300)

for i in range(10):
    yhat = model1.predict(input)
    input = np.roll(input, -1, axis=1)
    input[0][9] = yhat

for j in range(10):
        print(word2vec.most_similar([input[0][j]], topn=1)[0][0])


word2vec.most_similar([yhat[0]], topn=5)[0][0]