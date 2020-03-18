from fashion_mnist import x_train_clr0, x_test_clr0, x_train_clr1, x_test_clr1
from fashion_mnist import y_train_clr0, y_test_clr0, y_train_clr1, y_test_clr1
from fashion_mnist import x_train, x_test
from fashion_mnist import y_train, y_test
from auto_encoder import auto_encoder
import numpy as np


########################################################################################################################

def get_shuffle_configuration(array):
    config = np.arange(array.shape[0])
    np.random.shuffle(config)
    return config


########################################################################################################################

# pass fashion_MNIST images with keys through the auto-encoder
encoded_x_train_clr0, encoded_x_test_clr0 = auto_encoder(x_train_clr0, x_train, x_test_clr0, x_test, '0')
encoded_x_train_clr1, encoded_x_test_clr1 = auto_encoder(x_train_clr1, x_train, x_test_clr1, x_test, '1')

# connect sets with clearance 0, 1 and no clearance to one data set
train_images = np.concatenate([encoded_x_train_clr0, encoded_x_train_clr1, x_train], axis=0)
train_labels = np.concatenate([y_train_clr0, y_train_clr1, y_train])
test_images = np.concatenate([encoded_x_test_clr0, encoded_x_test_clr1, x_test], axis=0)
test_labels = np.concatenate([y_test_clr0, y_test_clr1, y_test])

# create shuffling configuration
train_config = get_shuffle_configuration(train_images)
test_config = get_shuffle_configuration(test_images)

# shuffle data sets
train_images = train_images[train_config]
train_labels = train_labels[train_config]
test_images = test_images[test_config]
test_labels = test_labels[test_config]

########################################################################################################################
