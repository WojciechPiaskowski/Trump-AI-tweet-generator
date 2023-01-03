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
from sklearn.preprocessing import StandardScaler

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
df['content'] = df['content'].apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'pic.twitter', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'.com', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'@', '', x))

# size of embeding dimension (it is set to 300 as that is the number of dimensions in google word2vec network)
embd_size = 300

# call google news word2vec model
# you can find it here: https://code.google.com/archive/p/word2vec/
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

all_words = []

for i in df['content']:
        for j in i.split():
            all_words.append(j.lower())

obs = 128*2000

all_words = all_words[:obs]

unique_words = pd.DataFrame(set(all_words), columns=['words'])
# get index unique_words[unique_words['words'] == 'hawaii.'].index[0]

all_words_emb = np.empty((len(all_words), embd_size))
j = 0
for i in all_words:
    try:
        all_words_emb[j] = word2vec[i]
        j += 1
    except:
        continue

lag = 3
N = all_words_emb.shape[0]-lag

X = np.empty((N, lag, embd_size))
Y = np.empty((N,))
j = 0
for i in range(N-1):
    x = all_words_emb[i:lag+i]
    y = unique_words[unique_words['words'] == str(all_words[lag+i])].index[0]
    X[j] = x
    Y[j] = int(y)
    j += 1

X_train = X[:int(obs/4*3)]
Y_train = Y[:int(obs/4*3)]
X_test = X[int(obs/4*3):]
Y_test = Y[int(obs/4*3):]

i = Input(shape=(lag, 300,))
x = LSTM(128, return_sequences=True)(i)
x = Dropout(0.5)(x)
x = LSTM(128)(x)
x = Dropout(0.5)(x)
x = Dense(len(unique_words), activation='softmax')(x)

cb = ES.EarlyStopping(monitor='val_loss', patience=10)
model1 = Model(i, x)
model1.compile(optimizer=Adam(lr=0.000001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
r1 = model1.fit(X_train, Y_train, epochs=30, validation_data=(X_test, Y_test), batch_size=64, callbacks=[cb])

plt.plot(r1.history['loss'], label='loss')
plt.plot(r1.history['val_loss'], label=['val_loss'])
plt.legend()

idx = np.random.randint(X.shape[0])
input = X[idx]
input = input.reshape(1, lag, 300)
tweet = ''

for i in range(30):
    yhat = model1.predict(input)
    yhat = np.argmax(yhat, axis=1)[0]
    yhat_word = unique_words.iloc[yhat][0]
    yhat_emb = word2vec[str(yhat_word)]
    input = np.roll(input, -1, axis=1)
    input[0][2] = yhat_emb
    tweet += ' ' + str(yhat_word)

print(tweet)

for j in range(3):
        print(word2vec.most_similar([input[0][j]], topn=1)[0][0])