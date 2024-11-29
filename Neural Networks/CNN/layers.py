import numpy as np

class ConvLayer:    
    def __init__(self, num_filters, filter_size, input_depth):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) * 0.1
        self.biases = np.zeros((num_filters, 1))

    def forward(self, x):
        self.input = x
        output_dim = x.shape[2] - self.filter_size + 1
        self.output = np.zeros((x.shape[0], self.num_filters, output_dim, output_dim))
        for b in range(x.shape[0]):
            for f in range(self.num_filters):
                for i in range(output_dim):
                    for j in range(output_dim):
                        region = x[b, :, i:i + self.filter_size, j:j + self.filter_size]
                        self.output[b, f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]
        return self.output

    def backward(self, d_output, learning_rate):
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)

        for b in range(d_output.shape[0]):
            for f in range(self.num_filters):
                for i in range(d_output.shape[2]):
                    for j in range(d_output.shape[3]):
                        region = self.input[b, :, i:i + self.filter_size, j:j + self.filter_size]
                        d_filters[f] += region * d_output[b, f, i, j]
                        d_input[b, :, i:i + self.filter_size, j:j + self.filter_size] += self.filters[f] * d_output[b, f, i, j]
                d_biases[f] += np.sum(d_output[:, f, :, :])

        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases
        return d_input


class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, x):
        self.input = x
        output_dim = x.shape[2] // self.pool_size
        self.output = np.zeros((x.shape[0], x.shape[1], output_dim, output_dim))
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                for i in range(output_dim):
                    for j in range(output_dim):
                        region = x[b, c, i * self.pool_size:(i + 1) * self.pool_size, j * self.pool_size:(j + 1) * self.pool_size]
                        self.output[b, c, i, j] = np.max(region)
        return self.output

    def backward(self, d_output):
        d_input = np.zeros_like(self.input)
        for b in range(d_output.shape[0]):
            for c in range(d_output.shape[1]):
                for i in range(d_output.shape[2]):
                    for j in range(d_output.shape[3]):
                        region = self.input[b, c, i * self.pool_size:(i + 1) * self.pool_size, j * self.pool_size:(j + 1) * self.pool_size]
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        d_input[b, c, i * self.pool_size + max_idx[0], j * self.pool_size + max_idx[1]] = d_output[b, c, i, j]
        return d_input


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
