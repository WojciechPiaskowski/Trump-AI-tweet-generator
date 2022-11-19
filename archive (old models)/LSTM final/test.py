# imports
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import re
from sklearn.utils import class_weight
from keras.losses import sparse_categorical_crossentropy
import csv
from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap
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
length = 70

i = Input(shape=(length, 1,))
x = LSTM(512, return_sequences=True)(i)
x = Dropout(0.5)(x)
x = LSTM(512)(x)
x = Dropout(0.5)(x)
x = Dense(n_vocab, activation='softmax')(x)

model1 = Model(i, x)


#class_weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
#class_weights_ann = dict(enumerate(class_weights))

#train_losses = []

# training

model1.load_weights('lag70_512_neurons.h5')

import sys
np.set_printoptions(threshold=sys.maxsize)