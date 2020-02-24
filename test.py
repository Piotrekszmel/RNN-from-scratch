from rnn import RNN
from preprocessing import generate_sequences_from_texts, generate_padded_sequences


with open("input.txt", "r") as f:
    sentences, vocab_size, tokenizer = generate_sequences_from_texts(f.read().split("\n\n"))
    predictors, labels, max_len = generate_padded_sequences(sentences, vocab_size)
    
print(predictors[:100], "\n")
print(labels[:100], "\n")
print(max_len)