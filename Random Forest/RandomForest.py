import numpy as np
from DecisionTree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=100, min_samples_split=2, num_features=None, max_depth=100):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.max_depth = max_depth
        self.estimators = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTree(min_sample_split=self.min_samples_split, num_features=self.num_features, max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.estimators.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
