import numpy as np
import pandas as pd

num_samples = 200

c1 = np.random.normal(loc=[2, 2], scale=[1, 1], size=(num_samples // 2, 2))
labels1 = np.zeros((num_samples // 2, ))

c2 = np.random.normal(loc=[7, 5], scale=[1, 1], size=(num_samples // 2, 2))
labels2 = np.ones((num_samples // 2, ))

features = np.vstack((c1, c2))
labels = np.concatenate((labels1, labels2))

df = pd.DataFrame({'feature1': features[:, 0], 'feature2': features[:, 1], 'label': labels})

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('dataset.csv', index=False)

print("The dataset is generated")
