import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from KNN import KNN


def train_test_split(data, test_size=0.2):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    X_train = data.iloc[train_indices][['feature1', 'feature2']].values
    y_train = data.iloc[train_indices]['label'].values
    X_test = data.iloc[test_indices][['feature1', 'feature2']].values
    y_test = data.iloc[test_indices]['label'].values

    return X_train, X_test, y_train, y_test


def evaluate(y_test, y_pred):
    accuracy = np.mean(y_pred == y_test)
    return f'Model Accuracy: {accuracy:.2%}'


data = pd.read_csv('dataset.csv')

plt.figure(figsize=(8, 6))
sns.scatterplot(x='feature1', y='feature2', hue='label', palette='viridis', data=data, s=80)
plt.title('Generated Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2)


model = KNN(k=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


test_point1 = np.array([[4, 3]])
test_point2 = np.array([[8, 7]])


def plot_knn_lines(test_point, neighbors, ax):
    for neighbor in neighbors:
        ax.arrow(test_point[0, 0], test_point[0, 1], neighbor[0] - test_point[0, 0], neighbor[1] - test_point[0, 1],
                 head_width=0.1, head_length=0.1, fc='gray', ec='gray')


predicted_class_test1 = model.predict(test_point1)[0]
predicted_class_test2 = model.predict(test_point2)[0]

distances_test1 = np.linalg.norm(X_train - test_point1, axis=1)
distances_test2 = np.linalg.norm(X_train - test_point2, axis=1)

k_nearest_indices_test1 = np.argsort(distances_test1)[:3]
k_nearest_neighbors_test1 = X_train[k_nearest_indices_test1]

k_nearest_indices_test2 = np.argsort(distances_test2)[:3]
k_nearest_neighbors_test2 = X_train[k_nearest_indices_test2]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='feature1', y='feature2', hue='label', data=data, palette='viridis', s=80)
plt.scatter(test_point1[0, 0], test_point1[0, 1], color='red', marker='X', s=100, label='Test Point')
plt.scatter(test_point2[0, 0], test_point2[0, 1], color='red', marker='X', s=100)

plot_knn_lines(test_point1, k_nearest_neighbors_test1, plt.gca())
plot_knn_lines(test_point2, k_nearest_neighbors_test2, plt.gca())

plt.title(f'KNN Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
print(evaluate(y_test, y_pred))

