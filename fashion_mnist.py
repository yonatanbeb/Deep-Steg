""" add the clearance levels to images and update labels """
from keras.datasets import fashion_mnist
from mnist import zeros, ones
from random import randrange
import numpy as np

# unload fashion_MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


########################################################################################################################

# layers key (mnist digit [0,1]) over mnist image
def clearance_level_zero(x, y):
    len0 = len(zeros)
    clearance_images = []
    clearance_labels = []
    for x, y in zip(x, y):
        key = zeros[randrange(len0)]
        clearance_images.append(x + key)
        # label for tops
        if y in [0, 2, 4, 6]:
            clearance_labels.append(10)
        # label for bottoms
        if y in [1, 3]:
            clearance_labels.append(11)
        # label for shoes
        if y in [5, 7, 9]:
            clearance_labels.append(12)
        # label for accessories (stays the same)
        if y == 8:
            clearance_labels.append(y)
    clearance_images = np.asarray(clearance_images, dtype=np.uint8)
    clearance_labels = np.asarray(clearance_labels, dtype=np.uint8)
    return clearance_images, clearance_labels


def clearance_level_one(x, y):
    len1 = len(ones)
    clearance_images = []
    clearance_labels = y
    for x in x:
        key = ones[randrange(len1)]
        clearance_images.append(x + key)
    clearance_images = np.asarray(clearance_images, dtype=np.uint8)
    clearance_labels = np.asarray(clearance_labels, dtype=np.uint8)
    return clearance_images, clearance_labels


########################################################################################################################

x_train_clr0, y_train_clr0 = clearance_level_zero(x_train, y_train)
x_test_clr0, y_test_clr0 = clearance_level_zero(x_test, y_test)

x_train_clr1, y_train_clr1 = clearance_level_one(x_train, y_train)
x_test_clr1, y_test_clr1 = clearance_level_one(x_test, y_test)

# no clearance data set doesn't return any information (13 => 'No Clearance')
x_train, y_train = x_train, np.ones(len(y_train)) * 13
x_test, y_test = x_test, np.ones(len(y_test)) * 13

########################################################################################################################
