import numpy as np

class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1 
        self.bh = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev):
        self.x = x
        self.h_prev = h_prev
        self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        return self.h

    def backward(self, dh_next, learning_rate):
        dtanh = dh_next * (1 - self.h ** 2)
        dWxh = np.dot(dtanh, self.x.T)
        dWhh = np.dot(dtanh, self.h_prev.T)
        dbh = np.sum(dtanh, axis=1, keepdims=True)
        dx = np.dot(self.Wxh.T, dtanh)
        dh_prev = np.dot(self.Whh.T, dtanh)

        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.bh -= learning_rate * dbh

        return dx, dh_prev


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, d_output, learning_rate):
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        d_input = np.dot(d_output, self.weights.T)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input
