import numpy as np

from preprocessing import load_Data
from rnn import Model


word_dim = 8000
hidden_dim = 100
np.random.seed(10)

X_train, Y_train = load_Data("data/reddit-comments-2015-08.csv")

rnn = Model(word_dim, hidden_dim)
losses = rnn.train(X_train[:100], Y_train[:100], learning_rate=0.005, num_epochs=10, evaluate_loss_after=1)

