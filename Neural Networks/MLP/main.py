import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def relu(Z):
    return np.maximum(0, Z)


def relu_deriv(Z):
    return Z > 0


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


class NeuralNetwork:
    def __init__(self, input_size, hidden_units=(5, 3), learning_rate=0.1, epochs=1000, batch_size=32):
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self._xavier_initializer()

    def _xavier_initializer(self):
        W1 = np.random.randn(self.hidden_units[0], self.input_size) * np.sqrt(2. / self.input_size)
        b1 = np.zeros((self.hidden_units[0], 1))
        W2 = np.random.randn(self.hidden_units[1], self.hidden_units[0]) * np.sqrt(2. / self.hidden_units[0])
        b2 = np.zeros((self.hidden_units[1], 1))
        W3 = np.random.randn(3, self.hidden_units[1]) * np.sqrt(2. / self.hidden_units[1])  # Output layer has 3 classes
        b3 = np.zeros((3, 1))
        return W1, b1, W2, b2, W3, b3

    def _forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = relu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = relu(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        A3 = softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    def _back_prop(self, Z1, A1, Z2, A2, Z3, A3, X, Y):
        m = X.shape[1]

        dZ3 = A3 - Y
        dW3 = (1 / m) * np.dot(dZ3, A2.T)
        db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.dot(self.W3.T, dZ3)
        dZ2 = dA2 * relu_deriv(Z2)
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * relu_deriv(Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    def _gradient_descent(self, dW1, db1, dW2, db2, dW3, db3):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

    def _compute_loss(self, A3, Y):
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(A3 + 1e-8)) / m
        return loss

    def fit(self, X, Y):
        for epoch in range(self.epochs):
            permutation = np.random.permutation(X.shape[1])
            X_shuffled, Y_shuffled = X[:, permutation], Y[:, permutation]
            for i in range(0, X.shape[1], self.batch_size):
                X_batch = X_shuffled[:, i:i + self.batch_size]
                Y_batch = Y_shuffled[:, i:i + self.batch_size]

                Z1, A1, Z2, A2, Z3, A3 = self._forward_prop(X_batch)
                dW1, db1, dW2, db2, dW3, db3 = self._back_prop(Z1, A1, Z2, A2, Z3, A3, X_batch, Y_batch)
                self._gradient_descent(dW1, db1, dW2, db2, dW3, db3)

            if epoch % 100 == 0:
                Z1, A1, Z2, A2, Z3, A3 = self._forward_prop(X)
                loss = self._compute_loss(A3, Y)
                predictions = self.predict(X)
                accuracy = np.mean(predictions == np.argmax(Y, axis=0)) * 100
                print(f"Epoch {epoch}: Loss {loss:.4f}, Accuracy {accuracy:.2f}%")

    def predict(self, X):
        _, _, _, _, _, A3 = self._forward_prop(X)
        predictions = np.argmax(A3, axis=0)
        return predictions


iris = load_iris()
X = iris.data.T
y = iris.target

X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1)).T

X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.3)

model = NeuralNetwork(input_size=X_train.shape[1], hidden_units=(8, 6), learning_rate=0.1, epochs=1000, batch_size=16)
model.fit(X_train.T, y_train.T)

y_pred = model.predict(X_test.T)
accuracy = np.mean(y_pred == np.argmax(y_test.T, axis=0)) * 100
print(f"Test Accuracy: {accuracy:.2f}%")
