import numpy as np
import itertools
import nltk
import csv 


def load_Data(path, vocabulary_size=8000):
    UNK_token = "UNK"
    START_token = "SENTENCE_START"
    END_token = "SENTENCE_END"
    
     # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["{} {} {}".format(START_token, x, END_token) for x in sentences]
    print("Parsed {} sentences".format(len(sentences)))
    
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Filter the sentences having few words (including SENTENCE_START and SENTENCE_END)
    tokenized_sentences = list(filter(lambda x: len(x) > 3, tokenized_sentences))
    
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found {} unique words tokens.".format(len(word_freq.items())))
    
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(UNK_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    
    print("Using vocabulary size {}.".format(vocabulary_size))
    print("The least frequent word in our vocabulary is '{}' and appeared {} times.".format(vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNK_token for w in sent]

    print("\nExample sentence: '{}'".format(sentences[1]))
    print("\nExample sentence after Pre-processing: '{}'\n".format(tokenized_sentences[0]))

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))

    # Print an training data example
    x_example, y_example = X_train[17], y_train[17]
    print("x:\n{}\n{}".format(" ".join([index_to_word[x] for x in x_example]), x_example))
    print("\ny:\n{}\n{}".format(" ".join([index_to_word[x] for x in y_example]), y_example))

    return X_train, y_train

if __name__ == '__main__':
    X_train, y_train = load_Data('data/reddit-comments-2015-08.csv')