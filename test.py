import pandas as pd

names = ['2010.csv', '2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv',
         '2017.csv', '2018.csv', '2019.csv', '2020.csv', '2021.csv', '2022.csv']

df = pd.DataFrame()
for file in names:
    df_tmp = pd.read_csv('elon_musk_tweets/' + file).loc[:, ['date', 'tweet']]
    
    df = pd.concat([df, df_tmp], axis=0)
    
df.shape




import random
import keras.callbacks
import keras_nlp.metrics
import numpy as np
import pandas as pd
import seaborn as sns
import re
import tensorflow as tf
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerDecoder
from keras.layers import TextVectorization

# from tensorflow.python.keras.layers import Input, Dense
# from tensorflow.python.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.optimizers import Adam


import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

# style config
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
sns.set_style('whitegrid')

# import Trump tweets (dataset from Kaggle.com, scrapped from twitter)
# df = pd.read_csv('realdonaldtrump.csv', sep=',')
# shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# first look at the data

# remove at least some links/unwanted strings
letters = set('abcdefghijklmnopqrstuvwxyz.,:?! ABCDEFGHIJKLMNOPQRSTUVWXYZ')

df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x))
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'pic.twitter', '', x))
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'.com', '', x))
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'@', '', x))
df['tweet'] = df['tweet'].apply(lambda x: ''.join(filter(letters.__contains__, x)))

print(f'total number of tweets: {df.shape[0]}')
print(f'total number of words: {df["tweet"].apply(lambda x: len(x.split())).sum()}')

# divide tweets into train, test sets (80/20).
# Validation set will be skipped, because there is less than 1M words in the dataset
length = len(df['tweet'])
tweets = list(df['tweet'])
tweets_train = list(tweets[: int(0.8 * length)])
tweets_test = list(tweets[int(0.8 * length):])

# get maximum length of a tweet
maxLen = df['tweet'].apply(lambda x: len(x.split())).max()


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
# TODO: change tf datasets to padnas/numpy or deeply understand tf datasets
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

# create the model

# hyper parameters
embed_dim = 256
num_head = 4
dropout = 0.3
epochs = 20
add_decoders = 2

def create_model():
    i = Input(shape=(maxLen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(vocab_size, maxLen, embed_dim)(i)
    decoder = TransformerDecoder(intermediate_dim=embed_dim, num_heads=num_head, dropout=dropout)(embedding_layer)

    for dec in range(add_decoders):
        decoder = TransformerDecoder(intermediate_dim=embed_dim, num_heads=num_head, dropout=dropout)(decoder)

    # x = Dense(128, activation='relu')(decoder)
    x = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(i, x)

    model.compile(optimizer=Adam(decay=0.001), loss='sparse_categorical_crossentropy',
                  metrics=[keras_nlp.metrics.Perplexity(), 'accuracy'])

    return model


model = create_model()
model.summary()  # TODO: summary output???


# callback
class TextSampler(keras.callbacks.Callback):
    def __init__(self, start_prompt, max_tokens):
        self.start_prompt = start_prompt
        self.max_tokens = max_tokens

    # method to choose a word based on highest probability
    def sample_token(self, logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype('int32')
        yhat = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        yhat = np.asarray(yhat).astype('float32')

        return np.random.choice(indices, p=yhat)

    def on_epoch_end(self, epoch, logs=None):
        decoded_sample = self.start_prompt

        for i in range(self.max_tokens - 1):
            tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
            yhat = self.model.predict([tokenized_prompt], verbose=1)
            sample_index = len(decoded_sample.strip().split()) - 1

            sampled_token = self.sample_token(yhat[0][sample_index])
            sampled_token = vocab_dict[sampled_token]
            decoded_sample += ' ' + sampled_token

        print(f'generated tweet:\n{decoded_sample}\n')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# first 5 words of a random tweet will be used as a seed / initial input

random_tweet = ' '.join(random.choice(tweets).split()[:4])
sampler = TextSampler(random_tweet, 30)
reducelr = keras.callbacks.ReduceLROnPlateau(patience=10, monitor='val_loss')

# model training

model = create_model()
history = model.fit(x_train, validation_data=x_test, epochs=epochs, callbacks=[sampler, reducelr, early_stopping])

# model inference

def sample_token(logits):

    logits, indices = tf.math.top_k(logits, k=5, sorted=True)
    indices = np.asarray(indices).astype('int32')
    yhat = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    yhat = np.asarray(yhat).astype('float32')

    return np.random.choice(indices, p=yhat)

def generate_tweet(prompt, tweet_length=20):

    generated_tweet = prompt

    for i in range(tweet_length - 1):

        tokenized_prompt = vectorize_layer([generated_tweet])[:, :-1]
        yhat = model.predict([tokenized_prompt], verbose=1)
        sample_index = len(generated_tweet.strip().split())-1

        sampled_token = sample_token(yhat[0][sample_index])
        sampled_token = vocab_dict[sampled_token]
        generated_tweet += ' ' + sampled_token

    return generated_tweet

print(generate_tweet('I think that hillary is', 50))
print(generate_tweet('What are those 5', 50))
print(generate_tweet('I think it is important that', 50))


# charts and evaluation
specs = f'dec: {add_decoders+1}, embed_dim: {embed_dim}, heads: {num_head}, drop: {dropout}, epochs: {epochs}'

with open('training_experiments_musk.txt', 'a+') as f:
    f.write(
f"""model: {specs},
val_loss: {np.round(history.history['val_loss'][-1], 2)},
min_val_loss: {np.round(np.min(history.history['val_loss']), 2)},
val_accuary: {np.round(history.history['val_accuracy'][-1], 2)},
val_perplexity: {np.round(history.history['val_perplexity'][-1], 1)} 
\n""")

plt.plot(np.array(history.history['loss']), label='loss')
plt.plot(np.array(history.history['val_loss']), label='val_loss')
plt.legend()

plt.plot(np.array(history.history['accuracy']), label='accuracy')
plt.plot(np.array(history.history['val_accuracy']), label='val_accuracy')
plt.legend()

plt.plot(np.array(history.history['perplexity']), label='perplexity')
plt.plot(np.array(history.history['val_perplexity']), label='val_perplexity')
plt.legend()
