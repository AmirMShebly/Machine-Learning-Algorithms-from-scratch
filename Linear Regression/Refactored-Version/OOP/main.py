from LinearRegression import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

model = LinearRegression()
model.train(df)

print("The parameters found by the optimization :  w =", model.w, "and b =", model.b)

plt.scatter(df.size_square_meter, df.price, color='blue')
plt.plot(list(range(75, 500)), [model.predict(x) for x in range(75, 500)], color='red')
plt.xlabel("size(square meters)")
plt.ylabel("price $")
plt.show()