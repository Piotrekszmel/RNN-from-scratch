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
            
    def calculate_total_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            self.forward(x[i])
            correct_word_predictions = [j.oout[y[i][k]] for k, j in enumerate(self.layers)]
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N
    
    def calculate_grads(self, x, y):
        for t, layer in enumerate(self.layers):
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            prev_s = np.zeros(self.hidden_dim)
            layer.backward(input, prev_s, y[t], self.U, self.W, self.V)
            prev_s = layer.sout
    
    def backward(self, x, y):
        self.forward(x)
        self.calculate_grads(x, y)
        dldu = np.zeros(self.U.shape)
        dldw = np.zeros(self.W.shape)
        dldv = np.zeros(self.V.shape)
        T = len(self.layers)
        for t in np.arange(T)[::-1]:
            dldv += np.outer(self.layers[t].doidv,self.layers[t].dldoi) # dim: [hidden_dim * word_dim]
            delta_t = np.matmul(self.layers[t].doidso, self.layers[t].dldoi) # dEt/dSt(out) dim: [hidden_dim] 
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                delta_t = delta_t * self.layers[bptt_step].dsodsi # dim: [hidden_dim]
                dldw += np.outer(self.layers[bptt_step].dsidw, delta_t)
                dldu += np.outer(self.layers[bptt_step].dsidu, delta_t)
                delta_t = np.matmul(self.layers[bptt_step].dsidpso, delta_t) # dim: [hidden_dim]
        return (dldu, dldw, dldv)
    
        
        