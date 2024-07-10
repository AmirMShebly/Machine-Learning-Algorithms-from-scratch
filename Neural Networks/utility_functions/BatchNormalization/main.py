import numpy as np
from BatchNormalization import BatchNormalization


X = np.random.randn(5, 3)

batch_norm_layer = BatchNormalization()

out = batch_norm_layer.forward(X, training=True)
print("Forward pass output:")
print(out)

dout = np.random.randn(5, 3)

dx, dgamma, dbeta = batch_norm_layer.backward(dout)
print("\nGradients:")
print("dx:", dx)
print("dgamma:", dgamma)
print("dbeta:", dbeta)