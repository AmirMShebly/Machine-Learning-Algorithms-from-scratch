import numpy as np
from Dropout import Dropout


X = np.random.randn(5, 3)

dropout = Dropout(rate=0.5)

output_train = dropout.forward(X, training=True)
print("Forward pass output during training:")
print(output_train)

output_test = dropout.forward(X, training=False)
print("\nForward pass output during testing:")
print(output_test)

dout = np.random.randn(5, 3)

dinput = dropout.backward(dout)
print("\nBackward pass output (gradients):")
print(dinput)
