import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from cnn.model import CNN
from cnn.utils import cross_entropy_loss

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = CNN()
epochs = 3
batch_size = 32
learning_rate = 0.01

for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        y_pred = model.forward(X_batch)
        loss = cross_entropy_loss(y_pred, y_batch)
        model.backward(y_pred, y_batch, learning_rate)

    y_test_pred = model.forward(X_test)
    accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
