import numpy as np


class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.iteration = 0

    def update(self, W, b, dw, db):
        self.iteration += 1

        # Update parameters
        W -= self.learning_rate * dw
        b -= self.learning_rate * db

        return W, b


def compute_gradients(W, b):
    dw = 2 * (W - 3)
    db = 2 * (b + 1)
    return dw, db


W = np.random.randn()
b = np.random.randn()

gd = GradientDescentOptimizer(learning_rate=0.1)

iterations = 1000

for i in range(iterations):
    dw, db = compute_gradients(W, b)
    W, b = gd.update(W, b, dw, db)

    if i % 100 == 0:
        loss = (W - 3) ** 2 + (b + 1) ** 2
        print(f"Iteration {i}: W = {W:.4f}, b = {b:.4f}, Loss = {loss:.4f}")

print(f"Final parameters: W = {W:.4f}, b = {b:.4f}")
