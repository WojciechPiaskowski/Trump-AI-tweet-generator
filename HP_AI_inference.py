# import libraries
import argparse
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization
from tensorflow.keras.models import load_model


# basic preprocessing function that will lowercase all words and replace 'next line' tags with space
def input_standardizer(input_string):
    sentence = tf.strings.lower(input_string)
    sentence = tf.strings.regex_replace(sentence, '\n', ' ')
    return sentence


if __name__ == "__main__":

    # arguments for running this script
    parser = argparse.ArgumentParser(description='Inference script for HP books AI')
    parser.add_argument('-k', '--k_predictions', type=int, default=3, required=False)
    parser.add_argument('-p', '--prompt', type=str, default='Harry looked to Hermione as if she', required=False)
    parser.add_argument('-w', '--n_words', type=int, default=40, required=False)
    args = parser.parse_args()

    # list of text files names
    names = ['Book1.txt', 'Book2.txt', 'Book3.txt', 'Book4.txt', 'Book5.txt', 'Book6.txt', 'Book7.txt']

    # join text into one string in full_text
    full_text = ''
    for file in names:
        filepath = 'data/HPbooks/' + file

        with open(filepath, encoding='utf-8') as f:
            text = f.read()
            full_text += text

    # split the text into sentences and cast it to a list
    full_text = full_text.split('.')
    full_text = list(filter(None, full_text))

    # shuffle the sentences dataset
    random.shuffle(full_text)
    maxLen = 50

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
    model = load_model('models/HP', compile=False)

    # print the model output
    print('HP books AI generated sentence text: \n')
    print(generate_sentence(args.prompt, args.n_words))