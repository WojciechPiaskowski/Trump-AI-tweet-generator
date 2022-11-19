import numpy as np
import pandas as pd
import seaborn as sns
import re
import tensorflow as tf
from tensorflow import keras
import keras_nlp
from tensorflow.keras.layers import TextVectorization

# style config
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
df.head()

# remove at least some links/unwanted strings
letters = set('abcdefghijklmnopqrstuvwxyz.,:?! ABCDEFGHIJKLMNOPQRSTUVWXYZ')

df['content'] = df['content'].apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'pic.twitter', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'.com', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'@', '', x))
df['content'] = df['content'].apply(lambda x: ''.join(filter(letters.__contains__, x)))

print(f'total number of tweets: {df.shape[0]}')
print(f'total number of words: {df["content"].apply(lambda x: len(x.split())).sum()}')

# divide tweets into train, test sets (80/20).
# Validation set will be skipped, because there is less than 1M words in the dataset
length = len(df['content'])
tweets = list(df['content'])
tweets_train = list(tweets[: int(0.8 * length)])
tweets_test = list(tweets[int(0.8 * length):])

# get maximum length of a tweet
maxLen = df['content'].apply(lambda x: len(x.split())).max()


def input_standardizer(input_string):
    sentence = tf.strings.lower(input_string)
    return sentence


# get vocabulary, its size and create a dictionary
# tokenize sentences and make all the words lower case
# make the tokenized input length equal to maximum length of tweets, apply padding if length is lower
vectorize_layer = TextVectorization(
    output_mode='int', output_sequence_length=int(maxLen + 1), standardize=input_standardizer)
vectorize_layer.adapt(tweets)
vocab = vectorize_layer.get_vocabulary()
vocab_size = len(vocab)
vocab_dict = dict(zip(range(vocab_size), vocab))

print(f'vocabulary size is {vocab_size}')

# create batched and shuffled datasets with tensorflow
# TODO: change tf datasets to padnas/numpy?
batch_size = 64

x_train = tf.data.Dataset.from_tensor_slices(tweets_train)
x_train = x_train.shuffle(buffer_size=256)
x_train = x_train.batch(batch_size)

x_test = tf.data.Dataset.from_tensor_slices(tweets_test)
x_test = x_test.shuffle(buffer_size=256)
x_test = x_test.batch(batch_size)


# preprocess the data using vectorize_layer and create x and y arrays
# y array is the target array, which is the next word in the tweet, based on previous words
def preprocess_tweets(tweet):
    tweet = tf.expand_dims(tweet, -1)
    tokenized_tweets = vectorize_layer(tweet)
    x = tokenized_tweets[:, :-1]
    y = tokenized_tweets[:, 1:]
    return x, y


tf.autograph.experimental.do_not_convert(preprocess_tweets)

x_train = x_train.map(tf.autograph.experimental.do_not_convert(preprocess_tweets))
x_train = x_train.prefetch(tf.data.AUTOTUNE)

x_test = x_test.map(tf.autograph.experimental.do_not_convert(preprocess_tweets))
x_test = x_test.prefetch(tf.data.AUTOTUNE)

# input word sequences are offset by 1
# batches are 64 tweets with 59 words (or padding) each
for entry in x_train.take(1):
    print(entry)
