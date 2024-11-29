import numpy as np
from .layers import ConvLayer, MaxPoolLayer, FullyConnectedLayer
from .activations import relu, softmax

class CNN:
    def __init__(self):
        self.conv1 = ConvLayer(8, 3, 1)
        self.pool1 = MaxPoolLayer(2)
        self.fc1 = FullyConnectedLayer(13 * 13 * 8, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = relu(x)
        x = self.pool1.forward(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc1.forward(x)
        return softmax(x)

    def backward(self, y_pred, y_true, learning_rate):
        d_output = y_pred - y_true
        d_output = self.fc1.backward(d_output, learning_rate)
        d_output = d_output.reshape(-1, 8, 13, 13)
        d_output = self.pool1.backward(d_output)
        self.conv1.backward(d_output, learning_rate)
