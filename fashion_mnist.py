""" add the clearance levels to images and update labels """
from keras.datasets import fashion_mnist
from mnist import zeros, ones
import numpy as np
from random import randrange

# unload fashion_MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# zero and one digit clearance keys
len0 = len(zeros)
len1 = len(ones)


# TODO: currently adds fashion_MNIST image and MNIST key on top of each other -- maybe need to append?
def clearance_level_zero(x, y):
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
    clearance_images = []
    clearance_labels = y
    for x in x:
        key = ones[randrange(len1)]
        clearance_images.append(x + key)
    clearance_images = np.asarray(clearance_images, dtype=np.uint8)
    clearance_labels = np.asarray(clearance_labels, dtype=np.uint8)
    return clearance_images, clearance_labels


# TODO: the images (not labels) need to pass through the AutoEncoder.
# TODO: find out if 'shuffle=True' supplements shuffling before
train_0 = clearance_level_zero(x_train, y_train)
test_0 = clearance_level_zero(x_test, y_test)

train_1 = clearance_level_one(x_train, y_train)
test_1 = clearance_level_one(x_test, y_test)

