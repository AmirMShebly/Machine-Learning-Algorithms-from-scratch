import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression


df = pd.read_csv("dataset.csv")
train_data = df.iloc[0:55]
X_train = np.array(train_data[['Hours_Studied', 'Hours_Project']])
y_train = np.array(train_data['Pass'])

model = LogisticRegression()
model.fit(X_train, y_train)

pass_data = df[df['Pass'] == 1]
fail_data = df[df['Pass'] == 0]

plt.figure(figsize=(8, 6))
plt.scatter(pass_data['Hours_Studied'], pass_data['Hours_Project'], color='g', label='Pass')
plt.scatter(fail_data['Hours_Studied'], fail_data['Hours_Project'], color='r', label='Fail')
plt.xlabel('Hours Studied')
plt.ylabel('Hours Project')
plt.legend()
plt.show()

w1, w2 = model.W[0][0], model.W[1][0]
print("The parameters found by the optimization: w1 =", w1, ", w2 =", w2, "and b =", model.b)

test_data = df.iloc[55:]
X_test = np.array(test_data[['Hours_Studied', 'Hours_Project']])
y_test = np.array(test_data['Pass'])

y_pred = model.predict(X_test)
accuracy = model.accuracy(y_pred, y_test)
print(f"Accuracy on the test data: {accuracy:.4f}")
