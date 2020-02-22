import numpy as np 
from utils import Sigmoid, Softmax


class Layer:
    def forward(self, X, prev, U, W, V):
        """ 
        X: input array
        prev: array
        U, W, V: weight matrices
        """
        
        activation = Sigmoid()
        output = Softmax()
        self.mul_U = np.matmul(X, U)
        self.mul_W = np.matmul(prev, W)
        self.sin = np.add(self.mul_U, self.mul_W)
        self.sout = activation(self.sin)
        self.oin = np.matmul(self.sout, V)
        self.oout = output.forward(self.oin)
    
    def backward(self, X, prev, y: Int, U, W, V):
        activation = Sigmoid()
        output = Softmax()
        self.loss = output.loss(self.oout, y)
        self.dldoi = output.backward(self.oout, y)
        self.doidso = V
        self.doidv = self.sout
        self.dsodsi = activation.backward(self.sout)
        self.dsidu = X
        self.dsidpso = W
        self.dsidw = prev