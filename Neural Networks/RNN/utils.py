import numpy as np

def cross_entropy_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]
