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
x = LSTM(512, return_sequences=True)(i)
x = Dropout(0.5)(x)
x = LSTM(512)(x)
x = Dropout(0.5)(x)
x = Dense(n_vocab, activation='softmax')(x)

model1 = Model(i, x)

# class_weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
# class_weights_ann = dict(enumerate(class_weights))

train_losses = []

# training

model1.load_weights('lag70_512_neurons.h5')

model1.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy')

for i in range(10):
    random_index = np.random.randint(0, len(X)-64)
    random_batch_X = X_train[random_index:random_index+64]
    random_batch_Y = Y_train[random_index:random_index+64]

    a = model1.train_on_batch(random_batch_X, random_batch_Y) #class_weight=class_weights_ann)
    train_losses.append(a)

    if i % 5000 == 0:
        train_losses = []
    if i % 1000 == 0:

        yhat = model1.predict(X_test)
        batch_loss = np.mean(sparse_categorical_crossentropy(Y_test, yhat))

        # save model
        model1.save('lag70_512_neurons.h5')
        print('train loss:', a)
        print('test loss:', batch_loss)
        print('train loss averaged:', np.mean(train_losses))


        # print example generated text (stochastic)
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

        with open('output.txt', 'a') as f:

            f.write('Batch number ' + str(i) + ' ' + ''.join(generated_text) + '\n')


        # print example generated text (deterministic)
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

        with open('output_deterministic.txt', 'a') as f:
            f.write('Batch number ' + str(i) + ' ' + ''.join(generated_text) + '\n')

        with open('losses.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i, batch_loss, a])


def generate_tweet(image_name, stochastic):

    if stochastic == True:
        start = np.random.randint(0, len(X_lst) - 1)
        seed_input_text = X_lst[start]
        seed_input = X[start].reshape(1, length, 1)
        generated_text_idx = []

        for j in range(280):
            yhat_prob = model1.predict(seed_input)
            yhat = np.random.choice(len(yhat_prob.reshape(n_vocab, )), p=yhat_prob.reshape(n_vocab, ))
            seed_input = np.roll(seed_input, -1)
            seed_input[:, -1] = yhat / n_vocab
            generated_text_idx.append(yhat)

        generated_text = [int_to_char[word] for word in generated_text_idx]
        generated_text = ''.join(generated_text)

    else:
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
        generated_text = ''.join(generated_text)

    # Constants
    # -----------------------------------------------------------------------------
    # Set the font to be used
    FONT_USER_INFO = ImageFont.truetype("arial.ttf", 50, encoding="utf-8")
    FONT_TEXT = ImageFont.truetype("arial.ttf", 70, encoding="utf-8")
    # Image dimensions (pixels)
    WIDTH = 2376
    HEIGHT = 2024
    # Color scheme
    COLOR_BG = 'white'
    COLOR_NAME = 'black'
    COLOR_TAG = (64, 64, 64)
    COLOR_TEXT = 'black'
    # Write coordinates
    COORD_PHOTO = (250, 170)
    COORD_NAME = (600, 185)
    COORD_TAG = (600, 305)
    COORD_TEXT = (250, 510)
    # Extra space to add in between lines of text
    LINE_MARGIN = 15
    # -----------------------------------------------------------------------------

    # Information for the image
    # -----------------------------------------------------------------------------
    user_name = "Trump AI bot"
    user_tag = "@Trump_AI_bot"
    text = str(generated_text)
    img_name = str(image_name)
    # -----------------------------------------------------------------------------

    # Setup of variables and calculations
    # -----------------------------------------------------------------------------
    # Break the text string into smaller strings, each having a maximum of 37\
    # characters (a.k.a. create the lines of text for the image)
    text_string_lines = wrap(text, 37)

    # Horizontal position at which to start drawing each line of the tweet body
    x = COORD_TEXT[0]

    # Current vertical position of drawing (starts as the first vertical drawing\
    # position of the tweet body)
    y = COORD_TEXT[1]

    # Create an Image object to be used as a means of extracting the height needed\
    # to draw each line of text
    temp_img = Image.new('RGB', (0, 0))
    temp_img_draw_interf = ImageDraw.Draw(temp_img)

    # List with the height (pixels) needed to draw each line of the tweet body
    # Loop through each line of text, and extract the height needed to draw it,\
    # using our font settings
    line_height = [
        temp_img_draw_interf.textsize(text_string_lines[i], font=FONT_TEXT)[1]
        for i in range(len(text_string_lines))
    ]
    # -----------------------------------------------------------------------------

    # Image creation
    # -----------------------------------------------------------------------------
    # Create what will be the final image
    img = Image.new('RGB', (WIDTH, HEIGHT), color='white')
    # Create the drawing interface
    draw_interf = ImageDraw.Draw(img)

    # Draw the user name
    draw_interf.text(COORD_NAME, user_name, font=FONT_USER_INFO, fill=COLOR_NAME)
    # Draw the user handle
    draw_interf.text(COORD_TAG, user_tag, font=FONT_USER_INFO, fill=COLOR_TAG)

    # Draw each line of the tweet body. To find the height at which the next\
    # line will be drawn, add the line height of the next line to the current\
    # y position, along with a small margin
    for index, line in enumerate(text_string_lines):
        # Draw a line of text
        draw_interf.text((x, y), line, font=FONT_TEXT, fill=COLOR_TEXT)
        # Increment y to draw the next line at the adequate height
        y += line_height[index] + LINE_MARGIN

    # Load the user photo (read-mode). It should be a 250x250 circle
    user_photo = Image.open('user_photo.png', 'r')

    # Paste the user photo into the working image. We also use the photo for\
    # its own mask to keep the photo's transparencies
    img.paste(user_photo, COORD_PHOTO, mask=user_photo)

    # Finally, save the created image
    img.save(f'{img_name}.png')

for i in range(30):
    sample = 'tweet_stochastic' + str(i)
    generate_tweet(sample, True)

for i in range(30):
    sample = 'tweet_deterministic' + str(i)
    generate_tweet(sample, False)