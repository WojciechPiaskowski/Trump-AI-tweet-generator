# import libraries
import random
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import keras.callbacks
import keras_nlp.metrics
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerDecoder
from keras.layers import TextVectorization
from tensorflow.keras.layers import Input, Dense  # imports not found in pycharm (autossugest doesn't work)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# select matplotlib backend to run charts in pycharm
mpl.use('Qt5Agg')

# style config
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
sns.set_style('whitegrid')

# import Trump tweets (dataset from Kaggle.com, scrapped from twitter)
df = pd.read_csv('data/realdonaldtrump.csv', sep=',')
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
full_text = list(df['content'])
full_text_train = list(full_text[: int(0.8 * length)])
full_text_test = list(full_text[int(0.8 * length):])

# get maximum length of a tweet
maxLen = df['content'].apply(lambda x: len(x.split())).max()


# preprocess function to lowercase all strings
def input_standardizer(input_string):
    sentence = tf.strings.lower(input_string)
    return sentence


# get vocabulary, its size and create a dictionary
# tokenize sentences and make all the words lower case
# make the tokenized input length equal to maxLen parameter, apply padding if length is lower
vectorize_layer = TextVectorization(
    output_mode='int', output_sequence_length=int(maxLen + 1), standardize=input_standardizer)
vectorize_layer.adapt(full_text)
vocab = vectorize_layer.get_vocabulary()
vocab_size = len(vocab)
vocab_dict = dict(zip(range(vocab_size), vocab))

print(f'vocabulary size is {vocab_size}')

# create batched and shuffled datasets with tensorflow datasets
batch_size = 64

x_train = tf.data.Dataset.from_tensor_slices(full_text_train)
x_train = x_train.shuffle(buffer_size=256)
x_train = x_train.batch(batch_size)

x_test = tf.data.Dataset.from_tensor_slices(full_text_test)
x_test = x_test.shuffle(buffer_size=256)
x_test = x_test.batch(batch_size)


# preprocess the data using vectorize_layer and create x and y arrays
# y array is the target array, which is the next word in the tweet, based on previous words
def preprocess_full_text(sentence):
    sentence = tf.expand_dims(sentence, -1)
    tokenized_full_text = vectorize_layer(sentence)
    x = tokenized_full_text[:, :-1]
    y = tokenized_full_text[:, 1:]
    return x, y


# preprocess train and test datasets with preprocess_full_text
x_train = x_train.map(preprocess_full_text)
x_test = x_test.map(preprocess_full_text)

# input and output word sequences are offset by 1
# batches are 64 sentences with maxLen words (or padding) each
for entry in x_train.take(1):
    print(entry)

# create the model
# hyperparameters that will be tuned with experiments
# high dropout layer due to training not enough data for a language model
embed_dim = 256
num_head = 4
dropout = 0.3
epochs = 20
add_decoders = 2


# transformer decoder based model architecture for next word prediction
# user perplexity as additional metric
def create_model():
    i = Input(shape=(maxLen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(vocab_size, maxLen, embed_dim)(i)
    decoder = TransformerDecoder(intermediate_dim=embed_dim, num_heads=num_head, dropout=dropout)(embedding_layer)

    for dec in range(add_decoders):
        decoder = TransformerDecoder(intermediate_dim=embed_dim, num_heads=num_head, dropout=dropout)(decoder)

    x = Dense(vocab_size, activation='softmax')(decoder)

    hp_model = Model(i, x)
    hp_model.compile(optimizer=Adam(decay=0.001), loss='sparse_categorical_crossentropy',
                     metrics=[keras_nlp.metrics.Perplexity(), 'accuracy'])

    return hp_model


# create the model and print a summary
model = create_model()
model.summary()


# function to choose a word based on probability distribution
# alternative approach would be to choose always the word with the highest probability
def sample_token(logits):
    logits, indices = tf.math.top_k(logits, k=5, sorted=True)
    indices = np.asarray(indices).astype('int32')
    yhat = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    yhat = np.asarray(yhat).astype('float32')

    return np.random.choice(indices, p=yhat)


# callback that creates an output sample after each epoch iteration
class TextSampler(keras.callbacks.Callback):
    def __init__(self, start_prompt, max_tokens):
        super().__init__()
        self.start_prompt = start_prompt
        self.max_tokens = max_tokens

    # on epoch end create a sample and decode it with vocab dictionary
    def on_epoch_end(self, epoch, logs=None):
        decoded_sample = self.start_prompt

        for i in range(self.max_tokens - 1):
            tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
            yhat = self.model.predict([tokenized_prompt], verbose=1)
            sample_index = len(decoded_sample.strip().split()) - 1

            sampled_token = sample_token(yhat[0][sample_index])
            sampled_token = vocab_dict[sampled_token]
            decoded_sample += ' ' + sampled_token

        print(f'generated sentence:\n{decoded_sample}\n')


# early stopping callback to cut the model before overfitting to the training data
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

# first 5 words of a random sentence will be used as a seed / initial input for sample generation
rand_sentence = ' '.join(random.choice(full_text).split()[:4])
sampler = TextSampler(rand_sentence, 30)
# callback that reduces learning rate if there is no improvement with loss
reducelr = keras.callbacks.ReduceLROnPlateau(patience=10, monitor='val_loss')

# model training
model = create_model()
history = model.fit(x_train, validation_data=x_test, epochs=epochs, callbacks=[sampler, reducelr, early_stopping])


# # model inference
def generate_sentence(prompt, sentence_length=20):
    generated_sentence = prompt

    for i in range(sentence_length - 1):
        tokenized_prompt = vectorize_layer([generated_sentence])[:, :-1]
        yhat = model.predict([tokenized_prompt], verbose=1)
        sample_index = len(generated_sentence.strip().split()) - 1

        sampled_token = sample_token(yhat[0][sample_index])
        sampled_token = vocab_dict[sampled_token]
        generated_sentence += ' ' + sampled_token

    return generated_sentence


# testing the output
print(generate_sentence('The focus should be on', 40))
print(generate_sentence('America is the only country', 40))
print(generate_sentence('Europe will need to think', 30))
print(generate_sentence('Voters showed the importance of', 30))

# save model hyperparameters and model metrics to a text file
specs = f'dec: {add_decoders + 1}, embed_dim: {embed_dim}, heads: {num_head}, drop: {dropout}, batch_size: {batch_size}'

with open('experiments\\training_experiments.txt', 'a+') as f:
    f.write(
        f"""model: {specs},
val_loss: {np.round(history.history['val_loss'][-1], 2)},
min_val_loss: {np.round(np.min(history.history['val_loss']), 2)},
val_accuracy: {np.round(history.history['val_accuracy'][-1], 2)},
val_perplexity: {np.round(history.history['val_perplexity'][-1], 1)},
\n""")

model.save('models/Trump')

# plots to understand the training during experimenting with hyperparameters

# plt.plot(np.array(history.history['loss']), label='loss')
# plt.plot(np.array(history.history['val_loss']), label='val_loss')
# plt.legend()
# plt.show()
#
# plt.plot(np.array(history.history['accuracy']), label='accuracy')
# plt.plot(np.array(history.history['val_accuracy']), label='val_accuracy')
# plt.legend()
# plt.show()
#
# plt.plot(np.array(history.history['perplexity']), label='perplexity')
# plt.plot(np.array(history.history['val_perplexity']), label='val_perplexity')
# plt.legend()
# plt.show()