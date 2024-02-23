import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from KMeans import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=1.0)


model = KMeans(n_clusters=3)
labels = model.fit(X)

score = silhouette_score(X, labels)
print("Silhouette Score:", score)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='x', label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
