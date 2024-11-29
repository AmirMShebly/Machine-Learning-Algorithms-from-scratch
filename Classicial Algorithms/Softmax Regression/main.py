import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report


class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=16):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None
        self.b = None

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y, y_hat):
        m = y.shape[0]
        log_likelihood = -np.log(y_hat[range(m), y.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def initialize_parameters(self, n_features, n_classes):
        self.W = np.random.randn(n_features, n_classes) * np.sqrt(2.0 / n_features)
        self.b = np.zeros((1, n_classes))

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        n_classes = y_train.shape[1]
        self.initialize_parameters(n_features, n_classes)

        for epoch in range(self.epochs):
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                Z = np.dot(X_batch, self.W) + self.b
                Y_hat = self.softmax(Z)

                loss = self.cross_entropy_loss(y_batch, Y_hat)

                dZ = Y_hat - y_batch
                dW = np.dot(X_batch.T, dZ) / X_batch.shape[0]
                db = np.sum(dZ, axis=0, keepdims=True) / X_batch.shape[0]

                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        Z = np.dot(X, self.W) + self.b
        Y_hat = self.softmax(Z)
        return np.argmax(Y_hat, axis=1)


def load_and_preprocess_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    return train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)


def main():

    X_train, X_test, y_train, y_test = load_and_preprocess_iris()

    nn_softmax = SoftmaxRegression(learning_rate=0.01, epochs=1000, batch_size=16)
    nn_softmax.fit(X_train, y_train)

    y_pred = nn_softmax.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred == y_test_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    print("Confusion Matrix:\n", confusion_matrix(y_test_labels, y_pred))
    print("\nClassification Report:\n", classification_report(y_test_labels, y_pred))


if __name__ == "__main__":
    main()

