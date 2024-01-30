import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        predictions = []

        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[indices]

            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]

            predictions.append(predicted_label)

        return np.array(predictions)

