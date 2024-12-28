import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        self.Wf = np.random.randn(hidden_size, input_size) * 0.1
        self.Uf = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bf = np.zeros((hidden_size, 1))

        # Input gate
        self.Wi = np.random.randn(hidden_size, input_size) * 0.1
        self.Ui = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))

        # Candidate memory
        self.Wc = np.random.randn(hidden_size, input_size) * 0.1
        self.Uc = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bc = np.zeros((hidden_size, 1))

        # Output gate
        self.Wo = np.random.randn(hidden_size, input_size) * 0.1
        self.Uo = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        # Forget gate
        f = self.sigmoid(np.dot(self.Wf, x) + np.dot(self.Uf, h_prev) + self.bf)

        # Input gate
        i = self.sigmoid(np.dot(self.Wi, x) + np.dot(self.Ui, h_prev) + self.bi)

        # Candidate memory
        c_tilde = np.tanh(np.dot(self.Wc, x) + np.dot(self.Uc, h_prev) + self.bc)

        # New memory cell
        c = f * c_prev + i * c_tilde

        # Output gate
        o = self.sigmoid(np.dot(self.Wo, x) + np.dot(self.Uo, h_prev) + self.bo)

        # New hidden state
        h = o * np.tanh(c)

        # Save for backpropagation
        self.x, self.h_prev, self.c_prev, self.f, self.i, self.c_tilde, self.c, self.o = x, h_prev, c_prev, f, i, c_tilde, c, o

        return h, c

    def backward(self, dh_next, dc_next, learning_rate):
        # Gradients of output gate
        do = dh_next * np.tanh(self.c)
        dWo = np.dot(do * self.o * (1 - self.o), self.x.T)
        dUo = np.dot(do * self.o * (1 - self.o), self.h_prev.T)
        dbo = np.sum(do * self.o * (1 - self.o), axis=1, keepdims=True)

        # Gradients of cell state
        dc = dh_next * self.o * (1 - np.tanh(self.c) ** 2) + dc_next
        df = dc * self.c_prev
        di = dc * self.c_tilde
        dc_tilde = dc * self.i

        # Gradients of forget gate
        dWf = np.dot(df * self.f * (1 - self.f), self.x.T)
        dUf = np.dot(df * self.f * (1 - self.f), self.h_prev.T)
        dbf = np.sum(df * self.f * (1 - self.f), axis=1, keepdims=True)

        # Gradients of input gate
        dWi = np.dot(di * self.i * (1 - self.i), self.x.T)
        dUi = np.dot(di * self.i * (1 - self.i), self.h_prev.T)
        dbi = np.sum(di * self.i * (1 - self.i), axis=1, keepdims=True)

        # Gradients of candidate memory
        dWc = np.dot(dc_tilde * (1 - self.c_tilde ** 2), self.x.T)
        dUc = np.dot(dc_tilde * (1 - self.c_tilde ** 2), self.h_prev.T)
        dbc = np.sum(dc_tilde * (1 - self.c_tilde ** 2), axis=1, keepdims=True)

        # Update weights
        self.Wf -= learning_rate * dWf
        self.Uf -= learning_rate * dUf
        self.bf -= learning_rate * dbf

        self.Wi -= learning_rate * dWi
        self.Ui -= learning_rate * dUi
        self.bi -= learning_rate * dbi

        self.Wc -= learning_rate * dWc
        self.Uc -= learning_rate * dUc
        self.bc -= learning_rate * dbc

        self.Wo -= learning_rate * dWo
        self.Uo -= learning_rate * dUo
        self.bo -= learning_rate * dbo

        # Gradients for previous timestep
        dx = np.dot(self.Wf.T, df) + np.dot(self.Wi.T, di) + np.dot(self.Wc.T, dc_tilde) + np.dot(self.Wo.T, do)
        dh_prev = np.dot(self.Uf.T, df) + np.dot(self.Ui.T, di) + np.dot(self.Uc.T, dc_tilde) + np.dot(self.Uo.T, do)
        dc_prev = f * dc

        return dx, dh_prev, dc_prev

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
