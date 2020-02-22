from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
import string

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


def clean_text(text):
    text = "".join(v for v in text if v not in string.punctuation).lower()
    text = text.encode("utf-8").decode("ascii", "ignore")
    return text


def generate_sequences_from_texts(texts):
    """
    Return tokenized n-grams of given texts, vocab_size and tokenizer
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    
    vocab_size = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in texts:
        tokens = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(tokens)):
            n_gram_sequence = tokens[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, vocab_size, tokenizer


def generate_padded_sequences(input_sequences, vocab_size):
    max_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_len, padding="pre"))
    predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
    labels = to_categorical(labels, num_classes=vocab_size)
    return predictors, labels, max_len
    
    
    

input_sequences, vocab_size, tokenizer = generate_sequences_from_texts(["I am cool", "You are not that kind"])
predictors, labels, max_len = generate_padded_sequences(input_sequences, vocab_size)

    