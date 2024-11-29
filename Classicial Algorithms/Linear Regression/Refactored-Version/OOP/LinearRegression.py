import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.00001, epochs=301):
        self.w = 0
        self.b = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def cost_function(self, data):
        m = len(data)
        cost = 0
        for i in range(m):
            x = data.iloc[i].size_square_meter
            y = data.iloc[i].price
            cost += ((self.w * x + self.b) - y) ** 2
        return cost / m

    def gradient_descent(self, data):
        m = len(data)
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            x = data.iloc[i].size_square_meter
            y = data.iloc[i].price

            dj_dw += (1 / m) * ((self.w * x + self.b) - y) * x
            dj_db += (1 / m) * ((self.w * x + self.b) - y)

        tmp_w = self.w - (self.learning_rate * dj_dw)
        tmp_b = self.b - (self.learning_rate * dj_db)

        self.w = tmp_w
        self.b = tmp_b

    def train(self, data):
        for i in range(self.epochs):
            if i % 100 == 0:
                print("epoch = ", i, 'mse cost = ', self.cost_function(data))
            self.gradient_descent(data)

    def predict(self, x):
        return self.w * x + self.b
