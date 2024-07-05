import numpy as np


class RMSPropOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, decay_rate=0.001):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.iteration = 0
        self.S_dw = 0
        self.S_db = 0
        self.decay_rate = decay_rate

    def update(self, W, b, dw, db):
        self.iteration += 1

        self.learning_rate = self.initial_learning_rate / (1 + self.decay_rate * self.iteration)

        self.S_dw = self.beta * self.S_dw + (1 - self.beta) * dw ** 2
        self.S_db = self.beta * self.S_db + (1 - self.beta) * db ** 2

        W -= self.learning_rate * dw / (np.sqrt(self.S_dw) + self.epsilon)
        b -= self.learning_rate * db / (np.sqrt(self.S_db) + self.epsilon)

        return W, b


def compute_gradients(W, b):
    dw = 2 * (W - 3)
    db = 2 * (b + 1)
    return dw, db


W = np.random.randn()
b = np.random.randn()

rmsprop = RMSPropOptimizer(learning_rate=0.1, decay_rate=0.01)

iterations = 1000

for i in range(iterations):
    dw, db = compute_gradients(W, b)
    W, b = rmsprop.update(W, b, dw, db)

    if i % 100 == 0:
        loss = (W - 3) ** 2 + (b + 1) ** 2
        print(f"Iteration {i}: W = {W:.4f}, b = {b:.4f}, Loss = {loss:.4f}, Learning Rate = {rmsprop.learning_rate:.6f}")

print(f"Final parameters: W = {W:.4f}, b = {b:.4f}")
