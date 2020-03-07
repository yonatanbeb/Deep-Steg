from fashion_mnist import train_0, train_1, test_0, test_1
from fashion_mnist import x_train, x_test
from fashion_mnist import y_train, y_test
from auto_encoder import auto_encoder
import numpy as np

zero_clearance_train, zero_clearance_test = auto_encoder(train_0[0], x_train, test_0[0], x_test)
one_clearance_train, one_clearance_test = auto_encoder(train_1[0], x_train, test_1[0], x_test)

x_train_clr0 = zero_clearance_train
y_train_clr0 = train_0[1]
x_test_clr0 = zero_clearance_test
y_test_clr0 = test_0[1]

x_train_clr1 = one_clearance_train
y_train_clr1 = train_1[1]
x_test_clr1 = one_clearance_test
y_test_clr1 = test_1[1]


# TODO: create data set where images are labeled one if they are encoded with one and zero otherwise

# TODO: shuffle data and split to train and test groups

x_train
y_train
x_test
y_test
