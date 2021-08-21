import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras
from collections import Counter
import pickle
import numpy as np
import random
import sys
import time
import io
import re
import argparse


class Vectorizer:

    def __init__(self, text, word_tokens, pristine_input, pristine_output):
        self.word_tokens = word_tokens
        self._pristine_input = pristine_input
        self._pristine_output = pristine_output

        tokens = self._tokenize(text)
        # print('corpus length:', len(tokens))
        token_counts = Counter(tokens)
        tokens = [x[0] for x in token_counts.most_common()]
        self._token_indices = {x: i for i, x in enumerate(tokens)}
        self._indices_token = {i: x for i, x in enumerate(tokens)}
        self.vocab_size = len(tokens)
        print('Vocab size:', self.vocab_size)

    def _tokenize(self, text):
        if not self._pristine_input:
            text = text.lower()
        if self.word_tokens:
            if self._pristine_input:
                return text.split()
            return Vectorizer.word_tokenize(text)
        return text

    def _detokenize(self, tokens):
        if self.word_tokens:
            if self._pristine_output:
                return ' '.join(tokens)
            return Vectorizer.word_detokenize(tokens)
        return ''.join(tokens)

    def vectorize(self, text):
        """Transforms text to a vector of integers"""
        tokens = self._tokenize(text)
        indices = []
        for token in tokens:
            if token in self._token_indices:
                indices.append(self._token_indices[token])
            else:
                print('Ignoring unrecognized token:', token)
        return np.array(indices, dtype=np.int32)

    def unvectorize(self, vector):
        """Transforms a vector of integers back to text"""
        tokens = [self._indices_token[index] for index in vector]
        return self._detokenize(tokens)

    @staticmethod
    def word_detokenize(tokens):
        regexes = [
            (re.compile(r'[ ]?\\n[ ]?'), r'\n'),
            (re.compile(r"\b(can)\s(not)\b"), r'\1\2'),
            (re.compile(r"\b(d)\s('ye)\b"), r'\1\2'),
            (re.compile(r"\b(gim)\s(me)\b"), r'\1\2'),
            (re.compile(r"\b(gon)\s(na)\b"), r'\1\2'),
            (re.compile(r"\b(got)\s(ta)\b"), r'\1\2'),
            (re.compile(r"\b(lem)\s(me)\b"), r'\1\2'),
            (re.compile(r"\b(mor)\s('n)\b"), r'\1\2'),
            (re.compile(r"\b(wan)\s(na)\b"), r'\1\2'),
            (re.compile(r"([^' ]) ('ll|'re|'ve|n't)\b"), r"\1\2"),
            (re.compile(r"([^' ]) ('s|'m|'d)\b"), r"\1\2"),
            (re.compile(r'[ ]?”'), r'"'),
            (re.compile(r'[ ]?--[ ]?'), r'--'),
            (re.compile(r'([\[\(\{\<]) '), r'\1'),
            (re.compile(r' ([\]\)\}\>])'), r'\1'),
            (re.compile(r'([\]\)\}\>]) ([:;,.])'), r'\1\2'),
            (re.compile(r"([^']) ' "), r"\1' "),
            (re.compile(r' ([?!\.])'), r'\1'),
            (re.compile(r'([^\.])\s(\.)([\]\)}>"\']*)\s*$'), r'\1\2\3'),
            (re.compile(r'([#$]) '), r'\1'),
            (re.compile(r' ([;%:,])'), r'\1'),
            (re.compile(r'(“)[ ]?'), r'"')
        ]

        text = ' '.join(tokens)
        for regexp, substitution in regexes:
            text = regexp.sub(substitution, text)
        return text.strip()

    @staticmethod
    def word_tokenize(text):
        regexes = [
            (re.compile(r'(\s)"'), r'\1 “ '),
            (re.compile(r'([ (\[{<])"'), r'\1 “ '),
            (re.compile(r'([:,])([^\d])'), r' \1 \2'),
            (re.compile(r'([:,])$'), r' \1 '),
            (re.compile(r'\.\.\.'), r' ... '),
            (re.compile(r'([;@#$%&])'), r' \1 '),
            (re.compile(r'([?!\.])'), r' \1 '),
            (re.compile(r"([^'])' "), r"\1 ' "),
            (re.compile(r'([\]\[\(\)\{\}\<\>])'), r' \1 '),
            (re.compile(r'--'), r' -- '),
            (re.compile(r'"'), r' ” '),
            (re.compile(r"([^' ])('s|'m|'d) "), r"\1 \2 "),
            (re.compile(r"([^' ])('ll|'re|'ve|n't) "), r"\1 \2 "),
            (re.compile(r"\b(can)(not)\b"), r' \1 \2 '),
            (re.compile(r"\b(d)('ye)\b"), r' \1 \2 '),
            (re.compile(r"\b(gim)(me)\b"), r' \1 \2 '),
            (re.compile(r"\b(gon)(na)\b"), r' \1 \2 '),
            (re.compile(r"\b(got)(ta)\b"), r' \1 \2 '),
            (re.compile(r"\b(lem)(me)\b"), r' \1 \2 '),
            (re.compile(r"\b(mor)('n)\b"), r' \1 \2 '),
            (re.compile(r"\b(wan)(na)\b"), r' \1 \2 '),
            (re.compile(r'\n'), r' \\n ')
        ]

        text = " " + text + " "
        for regexp, substitution in regexes:
            text = regexp.sub(substitution, text)
        return text.split()


def _create_sequences(vector, seq_length, seq_step):
    passes = []
    for offset in range(0, seq_length, seq_step):
        pass_samples = vector[offset:]
        num_pass_samples = pass_samples.size // seq_length
        pass_samples = np.resize(pass_samples,
                                 (num_pass_samples, seq_length))
        passes.append(pass_samples)
    return np.concatenate(passes)


def  shape_for_stateful_rnn(data, batch_size, seq_length, seq_step):
    inputs = data[:-1]
    targets = data[1:]

    inputs = _create_sequences(inputs, seq_length, seq_step)
    targets = _create_sequences(targets, seq_length, seq_step)

    inputs = _batch_sort_for_stateful_rnn(inputs, batch_size)
    targets = _batch_sort_for_stateful_rnn(targets, batch_size)

    targets = targets[:, :, np.newaxis]
    return inputs, targets


def _batch_sort_for_stateful_rnn(sequences, batch_size):
    num_batches = sequences.shape[0] // batch_size
    num_samples = num_batches * batch_size
    reshuffled = np.zeros((num_samples, sequences.shape[1]), dtype=np.int32)
    for batch_index in range(batch_size):
        slice_start = batch_index * num_batches
        slice_end = slice_start + num_batches
        index_slice = sequences[slice_start:slice_end, :]
        reshuffled[batch_index::batch_size, :] = index_slice
    return reshuffled


def load_data(data_file, word_tokens, pristine_input, pristine_output, batch_size, seq_length=50, seq_step=25):
    global vectorizer

    try:
        with open(data_file, encoding='utf-8') as input_file:
            text = input_file.read()
    except FileNotFoundError:
        print("No input.txt in data_dir")
        sys.exit(1)

    skip_validate = True
    # try:
    #     with open(os.path.join(data_dir, 'validate.txt'), encoding='utf-8') as validate_file:
    #         text_val = validate_file.read()
    #         skip_validate = False
    # except FileNotFoundError:
    #     pass  # Validation text optional
    # self.seeds = find_random_seeds(text)
    all_text = text if skip_validate else '\n'.join([text, text_val])

    vectorizer = Vectorizer(all_text, word_tokens, pristine_input, pristine_output)
    data = vectorizer.vectorize(text)
    x, y = shape_for_stateful_rnn(data, batch_size, seq_length, seq_step)
    print("Word_tokens:", word_tokens)
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)

    if skip_validate:
        return x, y, None, None, vectorizer

    data_val = vectorizer.vectorize(text_val)
    x_val, y_val = shape_for_stateful_rnn(data_val, batch_size,
                                          seq_length, seq_step)
    print('x_val.shape:', x_val.shape)
    print('y_val.shape:', y_val.shape)
    return x, y, x_val, y_val, vectorizer


def make_model(batch_size, vocab_size, embedding_size=64, rnn_size=128, num_layers=2):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, batch_input_shape=(batch_size, None)))
    for layer in range(num_layers):
        model.add(LSTM(rnn_size, stateful=True, return_sequences=True))
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def train(model, x, y, x_val, y_val, batch_size, num_epochs):
    print('Training...')
    # print("Shape:", x.shape, y.shape)
    # print(num_epochs, batch_size, x[0], y[0])
    train_start = time.time()
    validation_data = (x_val, y_val) if (x_val is not None) else None
    callbacks = None
    model.fit(x, y, validation_data=validation_data,
                    batch_size=batch_size,
                    shuffle=False,
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks)
    # self.update_sample_model_weights()
    train_end = time.time()
    print('Training time', train_end - train_start)


def sample_preds(preds, temperature=1.0):
    preds = np.asarray(preds).astype(np.float64)
    preds += np.finfo(np.float64).tiny
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(model, vectorizer, seed, length=100, diversity=0.5):
    seed_vector = vectorizer.vectorize(seed)

    print("Seed:", seed, end=' ' if vectorizer.word_tokens else '')
    model.reset_states()
    preds = None
    for char_index in np.nditer(seed_vector):
        preds = model.predict(np.array([[char_index]]), verbose=0)

    sampled_indices = []
    for i in range(length):
        char_index = 0
        if preds is not None:
            char_index = sample_preds(preds[0][0], diversity)
        sampled_indices.append(char_index)
        preds = model.predict(np.array([[char_index]]), verbose=0)
    sample = vectorizer.unvectorize(sampled_indices)
    return sample



if __name__ == "__main__":

    batch_size = 32
    num_epochs = 10
    out_len = 200 
    seq_length = 50 
    use_words = False  
    data_file = "text_habr.txt" 
    seed = "A"  

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", action="store", required=False, dest="text", help="Input text file")
    parser.add_argument("-e", "--epochs", action="store", required=False, dest="epochs", help="Number of training epochs")
    parser.add_argument("-p", "--phrase_len", action="store", required=False, dest="phrase_len", help="Phrase analyse length")
    parser.add_argument("-o", "--out_len", action="store", required=False, dest="out_len", help="Output text length")
    parser.add_argument("-g", "--generate", action="store_true", required=False, dest='generate', help="Generate output only without training")
    args = parser.parse_args()
    if args.text is not None:
        data_file = args.text
    if args.epochs is not None:
        num_epochs = int(args.epochs)
    if args.phrase_len is not None:
        seq_length = int(args.phrase_len)
    if args.out_len is not None:
        out_len = int(args.out_len)


    pristine_input, pristine_output = False, False
    x, y, x_val, y_val, vectorizer = load_data(data_file, use_words, pristine_input, pristine_output, batch_size, seq_length)

    model_file = data_file.lower().replace('.txt', '.h5')

    if args.generate is False:
        model = make_model(batch_size, vectorizer.vocab_size)
        train(model, x, y, x_val, y_val, batch_size, num_epochs)
        model.save(filepath=model_file)

    model = keras.models.load_model(model_file)
    predict_model = make_model(1, vectorizer.vocab_size)
    predict_model.set_weights(model.get_weights())

    res = generate(predict_model, vectorizer, seed=seed, length=out_len)
    print(res)
