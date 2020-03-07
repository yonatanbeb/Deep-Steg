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

x_train
y_train
x_test
y_test


image_train = np.concatenate([x_train_clr0, x_train_clr1, x_train], axis=0)
label_train = np.concatenate([y_train_clr0, y_train_clr1, y_train])

image_test = np.concatenate([x_test_clr0, x_test_clr1, x_test], axis=0)
label_test = np.concatenate([y_test_clr0, y_test_clr1, y_test])


def get_shuffle_configuration(array):
    config = np.arange(array.shape[0])
    np.random.shuffle(config)
    return config


train_config = get_shuffle_configuration(image_train)
test_config = get_shuffle_configuration(image_test)

image_train = image_train[train_config]
label_train = label_train[train_config]

image_test = image_test[test_config]
label_test = label_test[test_config]

