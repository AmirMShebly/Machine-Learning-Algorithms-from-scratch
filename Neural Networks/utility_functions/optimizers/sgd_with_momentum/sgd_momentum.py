import numpy as np


class SGDMomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.iteration = 0
        self.velocity_W = 0
        self.velocity_b = 0

    def update(self, W, b, dw, db):
        self.iteration += 1

        # Update velocities
        self.velocity_W = self.momentum * self.velocity_W - self.learning_rate * dw
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * db

        # Update parameters
        W += self.velocity_W
        b += self.velocity_b

        return W, b


def compute_gradients(W, b):
    dw = 2 * (W - 3)
    db = 2 * (b + 1)
    return dw, db


W = np.random.randn()
b = np.random.randn()

sgd_momentum = SGDMomentumOptimizer(learning_rate=0.1, momentum=0.9)

iterations = 1000

for i in range(iterations):
    dw, db = compute_gradients(W, b)
    W, b = sgd_momentum.update(W, b, dw, db)

    if i % 100 == 0:
        loss = (W - 3) ** 2 + (b + 1) ** 2
        print(f"Iteration {i}: W = {W:.4f}, b = {b:.4f}, Loss = {loss:.4f}")

print(f"Final parameters: W = {W:.4f}, b = {b:.4f}")
