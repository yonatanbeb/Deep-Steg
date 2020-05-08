""" encodes fashion_mnist for model training in auto_encode_dataset """
from keras.datasets import mnist, fashion_mnist
import numpy as np
from random import randrange

########################################################################################################################
""" retrieve 0s and 1s from mnist to encode fashion_mnist """

# unload mnist
(digit_images, digit_labels), (_, _) = mnist.load_data()
zeros, ones = [], []

for i in range(len(digit_images)):
    if digit_labels[i] == 0:
        zeros.append(digit_images[i])
    elif digit_labels[i] == 1:
        ones.append(digit_images[i])

# numpy array of all 0s in mnist
zeros = np.asarray(zeros, dtype=np.uint8)
# numpy array of all 1s in mnist
ones = np.asarray(ones, dtype=np.uint8)


########################################################################################################################
""" encoding algorithm """


def encode_0(x, y):
    """
    encodes fashion_mnist, with mnist images of the digit 0, by performing matrix addition
    and updating the labels to the general labels: Top, Bottom, and Shoe.
    INPUT:
        x - numpy array of fashion_mnist images
        y - numpy array of labels for x
    OUTPUT:
        images - numpy array of (fashion_mnist image + mnist digit 0)s
        labels - numpy array of updated labels for x:
                 T-Shirt, Pullover, Coat, Shirt ==> Top
                 Trouser, Dress ==> Bottom
                 Sandal, Sneaker, Boot ==> Shoe
                 Bag ==> Bag
    """
    len0 = len(zeros)
    images, labels = [], []
    for x, y in zip(x, y):
        # generate random 0 to encode image
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
    # convert images, labels array to numpy array
    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    return images, labels


def encode_1(x, y):
    """
    encodes fashion_mnist, with mnist images of the digit 1, by performing matrix addition
        INPUT:
            x - numpy array of fashion_mnist images
            y - numpy array of labels for x
        OUTPUT:
            images - numpy array of (fashion_mnist image + mnist digit 1)s
            labels - doesn't change
        """
    len1 = len(ones)
    images, labels = [], y
    for x in x:
        # generate random 0 to encode image
        key = ones[randrange(len1)]
        images.append(x + key)
    # convert images, labels array to numpy array
    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    return images, labels


########################################################################################################################
""" encode fashion_mnist with clearance levels 0 and 1 """

# unload fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# encode fashion_mnist with clearance level 0
encoded_x_train_clr0, encoded_y_train_clr0 = encode_0(x_train, y_train)
encoded_x_test_clr0, encoded_y_test_clr0 = encode_0(x_test, y_test)

# encode fashion_mnist with clearance level 1
encoded_x_train_clr1, encoded_y_train_clr1 = encode_1(x_train, y_train)
encoded_x_test_clr1, encoded_y_test_clr1 = encode_1(x_test, y_test)

# original (=non-encoded) dataset has label 13 (='No Clearance')
x_train, y_train = x_train, np.ones(len(y_train)) * 13
x_test, y_test = x_test, np.ones(len(y_test)) * 13

########################################################################################################################
