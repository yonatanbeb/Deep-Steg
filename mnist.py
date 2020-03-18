""" get MNIST digits to use as encoding keys for fashion_MNIST clearance """
from keras.datasets import mnist
import numpy as np

# unload MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

zeros, ones = [], []
# iterate through MNIST and pick out 0s and 1s
for i in range(len(x_train)):
    if y_train[i] == 0:
        zeros.append(x_train[i])
    if y_train[i] == 1:
        ones.append(x_train[i])

# convert arrays of 0s and 1s to numpy arrays
zeros = np.asarray(zeros, dtype=np.uint8)
ones = np.asarray(ones, dtype=np.uint8)
