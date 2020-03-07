from fashion_mnist import train_0, train_1, test_0, test_1
from fashion_mnist import x_train, x_test
from fashion_mnist import y_train, y_test
from auto_encoder import auto_encoder
import numpy as np

auto_encoded_0 = auto_encoder(train_0[0], x_train, test_0[0], x_test)
auto_encoded_1 = auto_encoder(train_1[0], x_train, test_1[0], x_test)

# TODO: reshape auto_encoded_0 / auto_encoded_1 to 28x28 (now 784)
# TODO: re-normalize values of array elements returned from auto-encoder

fashion_MNIST_clearance_0 = {'x': np.append(auto_encoded_0[0], auto_encoded_0[1], axis=0),
                             'y': np.append(train_0[1], test_0[1])}

fashion_MNIST_clearance_1 = {'x': np.append(auto_encoded_1[0], auto_encoded_1[1], axis=0),
                             'y': np.append(train_1[1], test_1[1])}

fashion_MNIST = {'x': np.append(x_train, x_test, axis=0), 'y_train': np.append(y_train, y_test)}

data = {'x': np.append(fashion_MNIST_clearance_0['x'], fashion_MNIST_clearance_1['x'], fashion_MNIST['x'], axis=0),
        'y': np.append(fashion_MNIST_clearance_0['y'], fashion_MNIST_clearance_1['y'], fashion_MNIST['y'], axis=0)}
