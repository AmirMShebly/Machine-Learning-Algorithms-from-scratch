import numpy as np
from .layers import RNNCell, FullyConnectedLayer
from .activations import softmax

class RNN:
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        self.sequence_length = sequence_length
        self.rnn_cell = RNNCell(input_size, hidden_size)
        self.fc = FullyConnectedLayer(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        self.hiddens = np.zeros((self.sequence_length, x.shape[0], self.hidden_size))
        h_prev = np.zeros((x.shape[0], self.hidden_size))
        for t in range(self.sequence_length):
            h_prev = self.rnn_cell.forward(x[:, t, :], h_prev)
            self.hiddens[t] = h_prev
        self.output = self.fc.forward(h_prev)
        return softmax(self.output)

    def backward(self, y_pred, y_true, learning_rate):
        d_output = y_pred - y_true
        d_hidden = self.fc.backward(d_output, learning_rate)

        dh_next = d_hidden
        for t in reversed(range(self.sequence_length)):
            dx, dh_next = self.rnn_cell.backward(dh_next, learning_rate)
