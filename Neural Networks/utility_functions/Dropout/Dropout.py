import numpy as np


class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
            return x * self.mask
        else:
            return x

    def backward(self, dout):
        return dout * self.mask