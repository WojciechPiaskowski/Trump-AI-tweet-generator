# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Embedding,\
    Conv1D, MaxPooling1D, GlobalMaxPool1D, Conv2D, Flatten, GlobalMaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model, load_model
import os
import nltk
from gensim.models import Word2Vec, KeyedVectors
import re
# nltk.download('punkt')

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

# check maximum number of words in order to padd the sequences (each observation to have same number of words)
df['no of words'] = df['content'].apply(lambda x: len(x.split()))
max = df['no of words'].max()

# padd the sequences using ',' as additional words if necessary
def padding(content, no_of_words):
    return str(content) + (max-no_of_words) * ' ,'

df['content_padded'] = df[['content', 'no of words']].apply(lambda x: padding(*x), axis=1)
X = df['content_padded'].values

# size of embeding dimension (it is set to 300 as that is the number of dimensions in google word2vec network)
embd_size = 300

# call google news word2vec model
# you can find it here: https://code.google.com/archive/p/word2vec/
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

# assign number of tweets N and number of words in a tweet J
N = len(X) # number of tweets
J = len(X[432].split()) # number of words (each tweet has the same number of words because it was padded)

# create an empty array by the size NxJxEmbd_size -> in this case its around 45000 x 60 x 300
embedded_X = np.empty((N, J, embd_size))

# populate the array
for i in range(N-1):
    for q in range(J-1):
        word = X[i].split()[q]
        try:
            embedded_X[i][q] = word2vec[word]
        except:
            embedded_X[i][q] = np.zeros(shape=embd_size,)

# set T as number of words times embedding dimension as the output of 1 tweet
T = J * embd_size

# dimensionality of latent space
latent_dim1 = 100

# define the generator model (noise generation as input for the actual model)
def build_generator(latent_dim1,):
    i = Input(shape=(1, 1, latent_dim1,))
    x = Dense(2048)(i)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=1024, kernel_size=(1), strides=2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=T, kernel_size=(1), strides=2)(x)
    model = Model(i, x)
    return model

# define the discriminating model
def build_discriminator(size1,):
    i = Input(shape=(1, 1, size1,))
    x = Conv2D(filters=1024, kernel_size=(1))(i)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=512, kernel_size=(1))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    return model

# compile both models

# build and compile the discriminator
discrimnator = load_model('/models/model_d.h5') # is only neccesery after first creating and sasving the model
discrimnator.trainable = True # is only neccesery after first creating and saving the model
#discrimnator = build_discriminator(T)  # no longer neccesery, as we are loading the model
discrimnator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.0001, beta_1=0.5),
    metrics=['accuracy']
)

#generator = build_generator(latent_dim1) # no longer neccesery, as we are loading the model
generator = load_model('/models/model_g.h5') # is only neccesery after first creating and sasving the model
# create an input to represent noise sample from latent space
z = Input(shape=(1, 1, latent_dim1,))
tweet = generator(z) # pass noise through generator to get a tweet

# make sure only generator is trained
discrimnator.trainable = False
fake_pred = discrimnator(tweet) # the true output is fake, but we label them real

# create combined model
combined_model = Model(z, fake_pred)
combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, beta_1=0.5))

# train the GAN

# config
batch_size = 32
epochs = 1
sample_period = 100 # every x steps generate and save some data

# create batch labels to use when calling train_on_batch
ones = np.ones(batch_size)
zeroes = np.zeros(batch_size)

# store the losses
d_losses = []
g_losses = []

# create a folder to store generated images
if not os.path.exists('gan_tweets'):
    os.makedirs('gan_tweets')

if not os.path.exists('models'):
    os.makedirs('models')

# a function to generate a grid of random samples from the generator and save them to a file, also saves the models
def sample_tweets(epoch):
    noise = np.random.randn(1, latent_dim1).reshape(1, 1, 1, latent_dim1)
    tweets = generator.predict(noise).reshape(J, embd_size)
    text = []
    for a in range(J-1):
        text.append(word2vec.most_similar([tweets[a]], topn=1)[0][0])
    with open(('gan_tweets/trump_bot_tweets' + str(epoch) + '.txt'), 'w+', encoding='utf-8') as textfile:
        for item in text:
            textfile.write(str(item) + ' ')
    discrimnator.save('/models/model_d.h5')
    combined_model.save('/models/model_c.h5')
    generator.save('/models/model_g.h5')


# main training loop
for epoch in range(epochs):
    ########## train discriminator
    # select a random batch of tweets
    idx = np.random.randint(0, X.shape[0], batch_size)
    real_tweets = embedded_X[idx].reshape(32, 1, 1, T)

    # generate fake tweets
    noise = np.random.randn(batch_size, latent_dim1).reshape(batch_size, 1, 1, latent_dim1)
    fake_tweets = generator.predict(noise)

    # train DISCRIMINATOR
    # both loss and accuracy are returned
    d_loss_real, d_acc_real = discrimnator.train_on_batch(real_tweets, ones)
    d_loss_fake, d_acc_fake = discrimnator.train_on_batch(fake_tweets, zeroes)
    d_loss = 0.5 * (d_loss_real + d_acc_fake)
    d_acc = 0.5 * (d_acc_real + d_acc_fake)

    # train GENERATOR
    for i in range(15):
        noise = np.random.randn(batch_size, latent_dim1).reshape(batch_size, 1, 1, latent_dim1)
        g_loss = combined_model.train_on_batch(noise, ones)

    # save the losses
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if epoch % 10 == 0:
        print(f'epoch: {epoch+1} / {epochs}, d_loss: {d_loss:.2f}, d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}')
    if epoch % sample_period == 0:
        sample_tweets(epoch)

# plot the losses
plt.plot(g_losses, label='g_losses')
plt.plot(d_losses, label='d_losses')
plt.legend()

# overall the models seems the generate mediocore results, the discriminator accuracy is very high
# it is to be expected as GANs are a lol better in generating pictures as opposed to tex sequences, there are other
# approaches that might give better results for that task

# however it was an interesting learning experience and a challenge
# few examples of generated tweets after thousands of epoch are saved in the notepad file
