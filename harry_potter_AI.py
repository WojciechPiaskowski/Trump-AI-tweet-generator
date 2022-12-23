# import libraries
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import keras.callbacks
import keras_nlp.metrics
import numpy as np
import tensorflow as tf
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerDecoder
from keras.layers import TextVectorization
# from tensorflow.python.keras.layers import Input, Dense
# from tensorflow.python.keras import Model
from tensorflow.keras.layers import Input, Dense  # TODO: try to fix those
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# select matplotlib backend to run charts in pycharm
mpl.use('Qt5Agg')

# list of text files names
names = ['Book1.txt', 'Book2.txt', 'Book3.txt', 'Book4.txt', 'Book5.txt', 'Book6.txt', 'Book7.txt']

# join text into one string in full_text
full_text = ''
for file in names:
    filepath = 'HPbooks/' + file

    with open(filepath, encoding='utf-8') as f:
        text = f.read()
        full_text += text

# split the text into sentences and cast it to a list
full_text = full_text.split('.')
full_text = list(filter(None, full_text))

# shuffle the sentences dataset
random.shuffle(full_text)

# TODO: how many sentences and words are there

# divide full_text into train, test sets (80/20).
# Validation set will be skipped, because there is less than 1M words in the dataset
length = len(full_text)
full_text_train = list(full_text[: int(0.8 * length)])
full_text_test = list(full_text[int(0.8 * length):])

# set maximum length of a sentence at 50
# more might cause issues with RAM, since matrices are getting exponentially larger
# TODO: check if that is true
maxLen = 50


# basic preprocessing function that will lowercase all words and replace next line tags with space
def input_standardizer(input_string):
    sentence = tf.strings.lower(input_string)
    sentence = tf.strings.regex_replace(sentence, '\n', ' ')
    return sentence


# get vocabulary, its size and create a dictionary
# tokenize sentences and make all the words lower case
# make the tokenized input length equal to maximum length of full_text, apply padding if length is lower
vectorize_layer = TextVectorization(
    output_mode='int', output_sequence_length=int(maxLen + 1), standardize=input_standardizer)
vectorize_layer.adapt(full_text)
vocab = vectorize_layer.get_vocabulary()
vocab_size = len(vocab)
vocab_dict = dict(zip(range(vocab_size), vocab))

print(f'vocabulary size is {vocab_size}')

# create batched and shuffled datasets with tensorflow datasets
# TODO: change tf datasets to padnas/numpy or deeply understand tf datasets
batch_size = 64

x_train = tf.data.Dataset.from_tensor_slices(full_text_train)
x_train = x_train.shuffle(buffer_size=256)
x_train = x_train.batch(batch_size)

x_test = tf.data.Dataset.from_tensor_slices(full_text_test)
x_test = x_test.shuffle(buffer_size=256)
x_test = x_test.batch(batch_size)


# preprocess the data using vectorize_layer and create x and y arrays
# y array is the target array, which is the next word in the tweet, based on previous words
# TODO: understand this
def preprocess_full_text(tweet):
    tweet = tf.expand_dims(tweet, -1)
    tokenized_full_text = vectorize_layer(tweet)
    x = tokenized_full_text[:, :-1]
    y = tokenized_full_text[:, 1:]
    return x, y


# preprocess train and test datasets with above function
# tf.autograph.experimental.do_not_convert(preprocess_full_text)
# x_train = x_train.map(tf.autograph.experimental.do_not_convert(preprocess_full_text))
x_train = x_train.map(preprocess_full_text)
# x_train = x_train.prefetch(tf.data.AUTOTUNE)

x_test = x_test.map(preprocess_full_text)
# x_test = x_test.map(tf.autograph.experimental.do_not_convert(preprocess_full_text))
# x_test = x_test.prefetch(tf.data.AUTOTUNE)

# input word sequences are offset by 1
# batches are 64 sentences with 59 words (or padding) each
# TODO check if that is true
for entry in x_train.take(1):
    print(entry)

# create the model
# hyperparameters that will be tuned with experiments
embed_dim = 256
num_head = 5
dropout = 0.5
epochs = 20
add_decoders = 3


# transformer decoder based model architecture for next word prediction
# user perplexity as additional metric
# TODO: check what is that measure
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


# callback that creates an output sample after each epoch iteration
class TextSampler(keras.callbacks.Callback):
    def __init__(self, start_prompt, max_tokens):
        self.start_prompt = start_prompt
        self.max_tokens = max_tokens

    # method to choose a word based on highest probability
    # alternative approach would be to choose based on probability distribution
    def sample_token(self, logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype('int32')
        yhat = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        yhat = np.asarray(yhat).astype('float32')

        return np.random.choice(indices, p=yhat)

    # on epoch end create a sample and decode it with vocab dictionary
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


# early stopping callback to cut the model before overfitting to the training data
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# first 5 words of a random sentence will be used as a seed / initial input
random_tweet = ' '.join(random.choice(full_text).split()[:4])
sampler = TextSampler(random_tweet, 30)
# TODO: what is reduce lr on plateu callback
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
        sample_index = len(generated_tweet.strip().split()) - 1

        sampled_token = sample_token(yhat[0][sample_index])
        sampled_token = vocab_dict[sampled_token]
        generated_tweet += ' ' + sampled_token

    return generated_tweet

# testing the output
print(generate_tweet('Harry came to the realisation that', 40))
print(generate_tweet('Ron and Harry became aware', 40))
print(generate_tweet('the truth ultimately is', 20))
print(generate_tweet('harry said loudly that he is', 10))

# save model hyperparameters and model metrics to a text file
specs = f'dec: {add_decoders + 1}, embed_dim: {embed_dim}, heads: {num_head}, drop: {dropout}, batch_size: {batch_size}'

with open('training_experiments_HP.txt', 'a+') as f:
    f.write(
        f"""model: {specs},
val_loss: {np.round(history.history['val_loss'][-1], 2)},
min_val_loss: {np.round(np.min(history.history['val_loss']), 2)},
val_accuary: {np.round(history.history['val_accuracy'][-1], 2)},
val_perplexity: {np.round(history.history['val_perplexity'][-1], 1)},
\n""")

# plots to understand the training during experimenting with hyperparameters
plt.plot(np.array(history.history['loss']), label='loss')
plt.plot(np.array(history.history['val_loss']), label='val_loss')
plt.legend()

plt.plot(np.array(history.history['accuracy']), label='accuracy')
plt.plot(np.array(history.history['val_accuracy']), label='val_accuracy')
plt.legend()

plt.plot(np.array(history.history['perplexity']), label='perplexity')
plt.plot(np.array(history.history['val_perplexity']), label='val_perplexity')
plt.legend()
