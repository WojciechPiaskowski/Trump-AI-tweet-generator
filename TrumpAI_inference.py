# import libraries
import argparse
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization
from tensorflow.keras.models import load_model


# preprocess function to lowercase all strings
def input_standardizer(input_string):
    sentence = tf.strings.lower(input_string)
    return sentence


if __name__ == "__main__":

    # arguments for running this script
    parser = argparse.ArgumentParser(description='Inference script for Trump AI')
    parser.add_argument('-k', '--k_predictions', type=int, default=3, required=False)
    parser.add_argument('-p', '--prompt', type=str, default='Today, the challenge we face', required=False)
    parser.add_argument('-w', '--n_words', type=int, default=40, required=False)
    args = parser.parse_args()

    # import Trump tweets (dataset from Kaggle.com, scrapped from twitter)
    df = pd.read_csv('data/realdonaldtrump.csv', sep=',')
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # remove at least some links/unwanted strings
    letters = set('abcdefghijklmnopqrstuvwxyz.,:?! ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    df['content'] = df['content'].apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x))
    df['content'] = df['content'].apply(lambda x: re.sub(r'pic.twitter', '', x))
    df['content'] = df['content'].apply(lambda x: re.sub(r'.com', '', x))
    df['content'] = df['content'].apply(lambda x: re.sub(r'@', '', x))
    df['content'] = df['content'].apply(lambda x: ''.join(filter(letters.__contains__, x)))

    full_text = list(df['content'])

    # get maximum length of a tweet
    maxLen = df['content'].apply(lambda x: len(x.split())).max()

    # get vocabulary, its size and create a dictionary
    # tokenize sentences and make all the words lower case
    # make the tokenized input length equal to maxLen parameter, apply padding if length is lower
    vectorize_layer = TextVectorization(
        output_mode='int', output_sequence_length=int(maxLen + 1), standardize=input_standardizer)
    vectorize_layer.adapt(full_text)
    vocab = vectorize_layer.get_vocabulary()
    vocab_size = len(vocab)
    vocab_dict = dict(zip(range(vocab_size), vocab))

    # sample a word from predicted k words, use estimate probabilities (normalized with softmax activation)
    def sample_token(logits):
        logits, indices = tf.math.top_k(logits, k=args.k_predictions, sorted=True)
        indices = np.asarray(indices).astype('int32')
        yhat = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        yhat = np.asarray(yhat).astype('float32')

        return np.random.choice(indices, p=yhat)

    # generate a sentence with sampled words and decode them via vectorize_layer
    def generate_sentence(prompt, sentence_length=20):

        generated_sentence = prompt

        for i in range(sentence_length - 1):
            tokenized_prompt = vectorize_layer([generated_sentence])[:, :-1]
            yhat = model.predict([tokenized_prompt], verbose=0)
            sample_index = len(generated_sentence.strip().split()) - 1

            sampled_token = sample_token(yhat[0][sample_index])
            sampled_token = vocab_dict[sampled_token]
            generated_sentence += ' ' + sampled_token

        return generated_sentence

    # load the model
    model = load_model('models/Trump', compile=False)

    # print the model output
    print('Trump AI generated tweet text: \n')
    print(generate_sentence(args.prompt, args.n_words))

