import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=700):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def fit(self, X_train, y_train):
        m = len(y_train)
        self.W = np.zeros((X_train.shape[1], 1))
        self.b = 0

        for epoch in range(self.epochs):
            Z = np.dot(self.W.T, X_train.T) + self.b
            A = self.sigmoid(Z)
            dZ = A - y_train
            dW = np.dot(X_train.T, dZ.T) / m
            db = np.sum(dZ) / m

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict(self, X):
        Z = np.dot(self.W.T, X.T) + self.b
        A = self.sigmoid(Z)
        return (A > 0.5).astype(int)

    def accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true)


