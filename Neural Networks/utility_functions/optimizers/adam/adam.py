import numpy as np


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=0.001):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iteration = 0
        self.V_dw = 0
        self.S_dw = 0
        self.V_db = 0
        self.S_db = 0
        self.decay_rate = decay_rate

    def update(self, W, b, dw, db):
        self.iteration += 1

        # learning rate decay
        self.learning_rate = self.initial_learning_rate / (1 + self.decay_rate * self.iteration)

        # Momentum
        self.V_dw = self.beta1 * self.V_dw + (1 - self.beta1) * dw
        self.V_db = self.beta1 * self.V_db + (1 - self.beta1) * db

        # RMSProp
        self.S_dw = self.beta2 * self.S_dw + (1 - self.beta2) * dw ** 2
        self.S_db = self.beta2 * self.S_db + (1 - self.beta2) * db ** 2

        # Bias correction
        V_dw_corrected = self.V_dw / (1 - self.beta1 ** self.iteration)
        V_db_corrected = self.V_db / (1 - self.beta1 ** self.iteration)
        S_dw_corrected = self.S_dw / (1 - self.beta2 ** self.iteration)
        S_db_corrected = self.S_db / (1 - self.beta2 ** self.iteration)

        # Update parameters
        W -= self.learning_rate * V_dw_corrected / (np.sqrt(S_dw_corrected) + self.epsilon)
        b -= self.learning_rate * V_db_corrected / (np.sqrt(S_db_corrected) + self.epsilon)

        return W, b


def compute_gradients(W, b):
    dw = 2 * (W - 3)
    db = 2 * (b + 1)
    return dw, db


W = np.random.randn()
b = np.random.randn()

adam = AdamOptimizer(learning_rate=0.1, decay_rate=0.01)

iterations = 1000

for i in range(iterations):
    dw, db = compute_gradients(W, b)
    W, b = adam.update(W, b, dw, db)

    if i % 100 == 0:
        loss = (W - 3) ** 2 + (b + 1) ** 2
        print(f"Iteration {i}: W = {W:.4f}, b = {b:.4f}, Loss = {loss:.4f}, Learning Rate = {adam.learning_rate:.6f}")

print(f"Final parameters: W = {W:.4f}, b = {b:.4f}")
