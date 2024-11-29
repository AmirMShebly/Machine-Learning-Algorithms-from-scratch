import random
import numpy as np
from sklearn.datasets import load_iris
from nb import NaiveBayes


def train_test_split(X, y, test_size=0.2, random_state=None):
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    if random_state is not None:
        random.seed(random_state)

    indices = list(range(len(X)))
    random.shuffle(indices)

    num_test_samples = int(test_size * len(X))

    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def confusion_matrix(y_test, y_pred):
    classes = np.unique(np.concatenate([y_test, y_pred]))

    matrix = {c: {c: 0 for c in classes} for c in classes}

    for actual, predicted in zip(y_test, y_pred):
        if actual in classes and predicted in classes:
            matrix[actual][predicted] += 1

    print("\t\t", end="")
    for c in classes:
        print(f"{c:^8}", end="")
    print("\n" + "-" * (8 * len(classes) + 7))
    for true_class in classes:
        print(f"{true_class:<6}", end="")
        for pred_class in classes:
            print(f"{matrix[true_class][pred_class]:^8}", end="")
        print("")

    return matrix


data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = NaiveBayes()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
confusion_matrix(y_test, y_pred)
