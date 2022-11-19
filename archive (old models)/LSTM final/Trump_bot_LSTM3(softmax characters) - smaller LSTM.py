# imports
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import re
import tensorflow.keras.callbacks as ES
import os, sys
from sklearn.utils import class_weight
from keras.losses import sparse_categorical_crossentropy
import csv


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
letters = set('abcdefghijklmnopqrstuvwxyz.,:?! ABCDEFGHIJKLMNOPQRSTUVWXYZ')

df['content'] = df['content'].apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'pic.twitter', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'.com', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'@', '', x))
df['content'] = df['content'].apply(lambda x: ''.join(filter(letters.__contains__, x)))

text = ''

for i in df['content']:
        for j in i.split():
            text += ' ' + j.lower()

chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(text)
n_vocab = len(chars)

length = 10
X_lst = []
Y = []

for i in range(0, n_chars - length, 1):
    x = text[i:i + length]
    y = text[i+length]
    X_lst.append([char_to_int[char] for char in x])
    Y.append(char_to_int[y])

print(len(X_lst))
X = np.reshape(X_lst, (len(X_lst), length, 1))

# normalize data
X = X / float(n_vocab)
Y = np.array(Y)

del(df)
#######################################

X_train = X[:int(len(X)-20000)]
Y_train = Y[:int(len(X)-20000)]
X_test = X[int(len(X)-20000):]
Y_test = Y[int(len(X)-20000):]

i = Input(shape=(length, 1,))
x = LSTM(128, return_sequences=True)(i)
x = Dropout(0.2)(x)
x = LSTM(128)(x)
x = Dropout(0.2)(x)
x = Dense(n_vocab, activation='softmax')(x)


model1 = Model(i, x)


class_weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
class_weights_ann = dict(enumerate(class_weights))

train_losses = []

# training

model1.load_weights('maly_model.h5')

model1.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy')

for i in range(400000):
    random_index = np.random.randint(0, len(X)-64)
    random_batch_X = X_train[random_index:random_index+64]
    random_batch_Y = Y_train[random_index:random_index+64]

    a = model1.train_on_batch(random_batch_X, random_batch_Y, class_weight=class_weights_ann)
    train_losses.append(a)

    if i % 5000 == 0:
        train_losses = []
    if i % 1000 == 0:

        yhat = model1.predict(X_test)
        batch_loss = np.mean(sparse_categorical_crossentropy(Y_test, yhat))

        # save model
        model1.save('maly_model.h5')
        print('train loss:', a)
        print('test loss:', batch_loss)
        print('train loss averaged:', np.mean(train_losses))

        # print example generated text
        start = np.random.randint(0, len(X_lst) - 1)
        seed_input_text = X_lst[start]
        seed_input = X[start].reshape(1, length, 1)
        generated_text_idx = []

        for j in range(100):
            yhat_prob = model1.predict(seed_input)
            yhat = np.random.choice(len(yhat_prob.reshape(n_vocab,)), p=yhat_prob.reshape(n_vocab,))
            seed_input = np.roll(seed_input, -1)
            seed_input[:, -1] = yhat / n_vocab
            generated_text_idx.append(yhat)

        generated_text = [int_to_char[word] for word in generated_text_idx]
        print(''.join(generated_text))

        with open('output_maly_stochastic.txt', 'a') as f:
            f.write('Batch number ' + str(i) + ' ' + ''.join(generated_text) + '\n')

        # print example generated text
        start = np.random.randint(0, len(X_lst) - 1)
        seed_input_text = X_lst[start]
        seed_input = X[start].reshape(1, length, 1)
        generated_text_idx = []

        for j in range(100):
            yhat_prob = model1.predict(seed_input)
            yhat = np.argmax(yhat_prob)
            seed_input = np.roll(seed_input, -1)
            seed_input[:, -1] = yhat / n_vocab
            generated_text_idx.append(yhat)

        generated_text = [int_to_char[word] for word in generated_text_idx]
        print(''.join(generated_text))

        with open('output_maly_deterministic.txt', 'a') as f:
            f.write('Batch number ' + str(i) + ' ' + ''.join(generated_text) + '\n')


        with open('losses_maly.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i, batch_loss, a])







