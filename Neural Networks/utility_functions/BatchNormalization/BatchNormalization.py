import numpy as np


class BatchNormalization:
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.cache = None

    def build(self, input_shape):
        self.gamma = np.ones(input_shape)
        self.beta = np.zeros(input_shape)
        self.running_mean = np.zeros(input_shape)
        self.running_var = np.zeros(input_shape)

    def forward(self, x, training=True):
        if self.gamma is None:
            self.build(x.shape[1])

        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.cache = (x, x_normalized, batch_mean, batch_var)
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout):
        x, x_normalized, batch_mean, batch_var = self.cache
        N, D = dout.shape

        dgamma = np.sum(dout * x_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)

        dx_normalized = dout * self.gamma

        dvar = np.sum(dx_normalized * (x - batch_mean) * -0.5 * np.power(batch_var + self.epsilon, -1.5), axis=0)

        dmean = np.sum(dx_normalized * -1.0 / np.sqrt(batch_var + self.epsilon), axis=0) + dvar * np.mean(
            -2.0 * (x - batch_mean), axis=0)

        dx = (dx_normalized / np.sqrt(batch_var + self.epsilon)) + (dvar * 2 * (x - batch_mean) / N) + (dmean / N)

        return dx, dgamma, dbeta

