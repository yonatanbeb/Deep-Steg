from keras.datasets import mnist, fashion_mnist
import numpy as np
from random import randrange

########################################################################################################################

(digit_images, digit_labels), (_, _) = mnist.load_data()
zeros, ones = [], []

for i in range(len(digit_images)):
    if digit_labels[i] == 0:
        zeros.append(digit_images[i])
    elif digit_labels[i] == 1:
        ones.append(digit_images[i])

zeros = np.asarray(zeros, dtype=np.uint8)
ones = np.asarray(ones, dtype=np.uint8)


########################################################################################################################

def encode_clearance_level_0(x, y):
    len0 = len(zeros)
    images, labels = [], []
    for x, y in zip(x, y):
        key = zeros[randrange(len0)]
        images.append(x + key)
        # label for tops
        if y in [0, 2, 4, 6]:
            labels.append(10)
        # label for bottoms
        if y in [1, 3]:
            labels.append(11)
        # label for shoes
        if y in [5, 7, 9]:
            labels.append(12)
        # label for accessories (stays the same)
        if y == 8:
            labels.append(y)
    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    return images, labels


def encode_clearance_level_1(x, y):
    len1 = len(ones)
    images, labels = [], y
    for x in x:
        key = ones[randrange(len1)]
        images.append(x + key)
    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    return images, labels


def encode_image_with_0(image):
    len0 = len(zeros)
    return image + zeros[randrange(len0)]


def encode_image_with_1(image):
    len1 = len(ones)
    return image + ones[randrange(len1)]


########################################################################################################################

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

encoded_x_train_clr0, encoded_y_train_clr0 = encode_clearance_level_0(x_train, y_train)
encoded_x_test_clr0, encoded_y_test_clr0 = encode_clearance_level_0(x_test, y_test)

encoded_x_train_clr1, encoded_y_train_clr1 = encode_clearance_level_1(x_train, y_train)
encoded_x_test_clr1, encoded_y_test_clr1 = encode_clearance_level_1(x_test, y_test)

x_train, y_train = x_train, np.ones(len(y_train)) * 13
x_test, y_test = x_test, np.ones(len(y_test)) * 13

########################################################################################################################
