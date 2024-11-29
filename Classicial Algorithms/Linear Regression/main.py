import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')


def cost_function(w, b, data):
    """
    This function calculates the mean squared error cost for our
    linear regression model(Although it's not necessary, because what we
    need is the partial derivative of this function which I've manually computed
    and used in gradient descent algorithm for optimization
    """

    m = len(data)
    cost = 0
    for i in range(m):

        x = data.iloc[i].size_square_meter
        y = data.iloc[i].price
        cost += ((w * x + b) - y) ** 2
    # cost += [((w * data.iloc[i].size_square_meters + b) - data.iloc[i].price) ** 2 for i in range(m)]
    return cost / m


def gradient_descent(w, b, data, learning_rate):
    """
    With this function we compute the partial derivatives of the cost function(J) inorder to
    minimize it by the aid of gradient descent algorithm
    """
    m = len(data)
    dj_dw = 0
    dj_db = 0

    for i in range(m):

        x = data.iloc[i].size_square_meter
        y = data.iloc[i].price

        dj_dw += (1 / m) * ((w * x + b) - y) * x
        dj_db += (1 / m) * ((w * x + b) - y)

    tmp_w = w - (learning_rate * dj_dw)
    tmp_b = b - (learning_rate * dj_db)

    w = tmp_w
    b = tmp_b

    return w, b


w = 0
b = 0
learning_rate = 0.00001
epochs = 301


for i in range(epochs):
    if i % 100 == 0:
        print("epoch = ", i, 'mse cost = ', cost_function(w, b, df))
    w, b = gradient_descent(w, b, df, learning_rate)


print("The parameters found by the optimization :  w = ", w, "and b = ", b)

plt.scatter(df.size_square_meter, df.price, color='blue')
plt.plot(list(range(75, 500)), [w * x + b for x in range(75, 500)], color='red')
plt.xlabel("size(square meters)")
plt.ylabel("price $")
plt.show()

