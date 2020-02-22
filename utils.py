import numpy as np 


class Sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward(self, x):
        return (1.0 - x) * x


class Softmax:
    def forward(self, x):
        exp_x = np.exp(-x)
        return exp_x / np.sum(exp_x)
    
    def loss(self, x, y):
        return np.log[x[y]]
    
    def backward(self, x, y):
        x[y] -= 1.0
        return x
    
