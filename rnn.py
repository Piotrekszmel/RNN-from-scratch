import numpy as np 
import random
import pickle
from pathlib import Path
from Layer import Layer

class RNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, word_dim))
        self.layers = []
    
    def forward(self, x):
        """
        x : array of integers (denoting one training example i.e. a sentence)
        """
        T = len(x)
        self.layers = []
        prev_s = np.zeros(self.hidden_dim)
        for t in range(T):
            layer = Layer()
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            layer.forward(input, prev_s, self.U, self.W, self.V)
            prev_s = layer.sout
            self.layers.append(layer)
    
    def generate(self, seed, num=100, k=10):
        text = []
        text.append(seed)
        prev_s = np.zeros(self.hidden_dim)
        for i in range(num):
            layer = Layer()
            input = np.zeros(self.word_dim)
            input[seed] = 1
            layer.forward(input, prev_s, self.U, self.W, self.V)
            prev_s = layer.sout
            temp = sorted(layer.oout, reverse=True)
            threshold = temp[k-1]
            top = [index for index, val in enumerate(layer.oout) if val>=threshold]
            seed = random.choice(top)
            text.append(seed)
        return text
            
    def load(self, filename):
        weightsFile = Path(filename)
        if weightsFile.exists:
            print("Loading weights...")
            with open(filename, "rb") as f:
                self.U, self.W, self.V = pickle.load(f)
                