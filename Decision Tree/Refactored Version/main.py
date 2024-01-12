import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTree()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


score = accuracy(y_test, y_pred)

print(score)

















