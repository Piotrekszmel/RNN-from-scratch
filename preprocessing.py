from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras.utils as ku
import string

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


def clean_text(text):
    text = "".join(v for v in text if v not in string.punctuation).lower()
    text = text.encode("utf-8").decode("ascii", "ignore")
    return text


def generate_sequences_from_texts(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    
    vocab_size = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in texts:
        tokens = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(tokens)):
            n_gram_sequence = tokens[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, vocab_size
