import numpy as np
from .layers import LSTMCell, FullyConnectedLayer
from .activations import softmax

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        self.sequence_length = sequence_length
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.fc = FullyConnectedLayer(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size = x.shape[0]
        self.hiddens = np.zeros((self.sequence_length, batch_size, self.hidden_size))
        h_prev = np.zeros((batch_size, self.hidden_size))
        c_prev = np.zeros((batch_size, self.hidden_size))

        for t in range(self.sequence_length):
            h_prev, c_prev = self.lstm_cell.forward(x[:, t, :], h_prev, c_prev)
            self.hiddens[t] = h_prev

        self.output = self.fc.forward(h_prev)
        return softmax(self.output)

    def backward(self, y_pred, y_true, learning_rate):
        d_output = y_pred - y_true
        d_hidden = self.fc.backward(d_output, learning_rate)

        dh_next = d_hidden
        dc_next = np.zeros_like(d_hidden)
        for t in reversed(range(self.sequence_length)):
            dx, dh_next, dc_next = self.lstm_cell.backward(dh_next, dc_next, learning_rate)
