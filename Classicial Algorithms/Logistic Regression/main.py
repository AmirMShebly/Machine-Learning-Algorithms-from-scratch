import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(w1, w2, b, X_train, y_train, learning_rate):
    m, n = X_train.shape
    w = np.array([w1, w2])
    dj_dw = np.array([0.0, 0.0])
    dj_db = 0
    for i in range(m):
        for j in range(n):
            dj_dw[j] += ((sigmoid(np.dot(X_train[i], w.T) + b)) - y_train[i]) * X_train[i, j]
        dj_db += sigmoid(np.dot(X_train[i], w.T) + b) - y_train[i]

    dj_dw[0] /= m
    dj_dw[1] /= m
    dj_db /= m

    tmp_w1 = w1 - (learning_rate * dj_dw[0])
    tmp_w2 = w2 - (learning_rate * dj_dw[1])
    tmp_b = b - (learning_rate * dj_db)

    w1 = tmp_w1
    w2 = tmp_w2
    b = tmp_b

    return w1, w2, b


def score(X, y, w1, w2, b):
    X_test = np.array(X)
    y_test = np.array(y)
    results = []
    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            y_pred = sigmoid(- w1 * X_test[i, 0] - w2 * X_test[i, 1] + b) > 0.5
            results.append(y_pred == y_test[i])

    counter = 0
    for i in range(len(results)):
        if results[i]:
            counter += 1
    return counter / len(results)


train_data = df[:80]
X_train = np.array(train_data[['Hours_Studied', 'Hours_Project']])
y_train = np.array(train_data['Pass'])

test_data = df[80:]
X_test = test_data[['Hours_Studied', 'Hours_Project']]
y_test = test_data['Pass']


w1 = 0
w2 = 0
b = 0
learning_rate = 0.1
epochs = 700
for i in range(epochs):
    if i % 100 == 0:
        print('epoch', i, 'w1 = ', w1, 'w2 = ', w2, 'b = ', b)
    w1, w2, b = gradient_descent(w1, w2, b, X_train, y_train, learning_rate)

print("The parameters found by the optimization :  w1 = ", w1, ", w2 = ", w2, "and b = ", b)


pass_data = df[df['Pass'] == 1]
fail_data = df[df['Pass'] == 0]
# Visualizing the datapoints
plt.figure(figsize=(8, 6))

plt.scatter(pass_data['Hours_Studied'], pass_data['Hours_Project'], color='g', label='Pass')
plt.scatter(fail_data['Hours_Studied'], fail_data['Hours_Project'], color='r', label='Fail')

plt.xlabel('Hours Studied')
plt.ylabel('Hours Project')
plt.legend()
plt.show()

print("score : ", score(X_test, y_test, w1, w2, b))
# 0.8

