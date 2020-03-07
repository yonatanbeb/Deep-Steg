from PIL import Image
from auto_encode_dataset import x_train
from auto_encode_dataset import train_1, x_train_clr1
from auto_encode_dataset import train_0, x_train_clr0
import numpy as np


def display_image(img_array):
    display = Image.fromarray(img_array)
    display.show()


def display_triplet_clr0(index):
    trip = np.concatenate([x_train[index], train_0[0][index], x_train_clr0[index]], axis=0)
    display_image(trip)


def display_triplet_clr1(index):
    trip = np.concatenate([x_train[index], train_1[0][index], x_train_clr1[index]], axis=0)
    display_image(trip)
