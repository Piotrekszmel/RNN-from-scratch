import numpy as np 


class Sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward(self, x):
        return (1.0 - x) * x
