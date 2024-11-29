import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer


def train_test_split(X, y, test_size):
    num_samples = X.shape[0]
    num_test_samples = int(test_size * num_samples)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_indices = indices[num_test_samples:]
    test_indices = indices[:num_test_samples]

    X_train = X[train_indices, :]
    X_test = X[test_indices, :]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def is_leaf_node(node):
    return node['value'] is not None


def extend_tree(X, y, depth=0, max_depth=100, min_samples_split=2, num_features=None):
    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))

    if depth >= max_depth or n_labels == 1 or n_samples < min_samples_split:
        leaf_value = most_common_label(y)
        return {'value': leaf_value}

    if num_features is None:
        num_features = n_features
    feature_idxs = np.random.choice(n_features, num_features, replace=False)
    best_feature, best_threshold = best_split(X, y, feature_idxs)

    left_idxs, right_idxs = split(X[:, best_feature], best_threshold)
    left = extend_tree(X[left_idxs, :], y[left_idxs], depth + 1, max_depth, min_samples_split, num_features)
    right = extend_tree(X[right_idxs, :], y[right_idxs], depth + 1, max_depth, min_samples_split, num_features)
    return {'feature': best_feature, 'threshold': best_threshold, 'left': left, 'right': right, 'value': None}


def best_split(X, y, feature_idxs):
    max_info_gain = 0
    split_idx, split_threshold = None, None

    for i in feature_idxs:
        X_column = X[:, i]
        thresholds = np.unique(X_column)

        for j in thresholds:
            info_gain = information_gain(y, X_column, j)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                split_idx = i
                split_threshold = j

    return split_idx, split_threshold


def information_gain(y, X_column, threshold):
    parent_entropy = entropy(y)

    left_idxs, right_idxs = split(X_column, threshold)

    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0

    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    entropy_left, entropy_right = entropy(y[left_idxs]), entropy(y[right_idxs])
    child_entropy = (n_l / n) * entropy_left + (n_r / n) * entropy_right

    return parent_entropy - child_entropy


def entropy(y):
    if len(y) != 0:
        p = len(y[y == 1]) / len(y)
        if p != 0 and p != 1:
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return 0


def split(X_column, split_threshold):
    left_idxs = np.argwhere(X_column <= split_threshold).flatten()
    right_idxs = np.argwhere(X_column > split_threshold).flatten()
    return left_idxs, right_idxs


def most_common_label(y):
    counter = Counter(y)
    value = counter.most_common(1)[0][0]
    return value


def traverse_tree(x, node):
    if is_leaf_node(node):
        return node['value']

    if x[node['feature']] <= node['threshold']:
        return traverse_tree(x, node['left'])
    return traverse_tree(x, node['right'])


def predict(X, tree):
    return np.array([traverse_tree(i, tree) for i in X])


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree = extend_tree(X_train, y_train, max_depth=10)

y_pred = predict(X_test, tree)

score = accuracy(y_test, y_pred)

print(f"Accuracy: {score:.2f}")
# 0.9 +/- 0.05
