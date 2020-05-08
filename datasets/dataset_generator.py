""" generate encoded and auto encoded fashion_mnist datasets with trained Auto Encoder models"""
from keras.datasets import fashion_mnist
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import sys
sys.path.append('../')
from encode_dataset import encode_0, encode_1

########################################################################################################################

MODEL_PATH = os.path.abspath('../models')

# unload fashion_mnist
(images, _), (_, _) = fashion_mnist.load_data()
# get the first 100 images from fashion_mnist
images = images[:100]


########################################################################################################################
""" function to auto encode with current existing Auto Encoder models """


def auto_encode(encoded_images, clearance_level):
    """
    INPUT:
        encoded_images: numpy array of fashion_mnist encoded with clearance level for training
        clearance_level [0, 1[: the number encoded into encoded_images
    OUTPUT:
        auto_encoded_images: encoded_images auto encoded with their clearance level
    """
    # load AutoEncoder models:
    Encoder = load_model(MODEL_PATH + '/encoder_' + str(clearance_level) + '.h5')
    Decoder = load_model(MODEL_PATH + '/decoder_' + str(clearance_level) + '.h5')

    # normalize and reshape encoded_images
    encoded_images = (encoded_images.astype('float32') / 255).reshape(encoded_images.shape[0], 784)

    # auto encode encoded_images
    encoded_images = Encoder.predict(encoded_images)
    auto_encoded_images = Decoder.predict(encoded_images)

    # re-normalize and re-shape auto_encoded_images
    auto_encoded_images = (auto_encoded_images * 255).astype('uint8').reshape(auto_encoded_images.shape[0], 28, 28)

    return auto_encoded_images

########################################################################################################################


# encode original images with 0
encoded_images_clr0, _ = encode_0(images, (np.ones(len(images))))
# encode original images with 1
encoded_images_clr1, _ = encode_1(images, (np.ones(len(images))))

# auto encode encoded_images_0
auto_encoded_images_clr0 = auto_encode(encoded_images_clr0, 0)
# auto encode encoded_images_1
auto_encoded_images_clr1 = auto_encode(encoded_images_clr1, 1)

# save encoded and auto_encoded images
for i in range(100):
    Image.fromarray(encoded_images_clr0[i]).save('encoded0/' + str(i) + '.png')
    Image.fromarray(encoded_images_clr1[i]).save('encoded1/' + str(i) + '.png')
    Image.fromarray(auto_encoded_images_clr0[i]).save('auto_encoded0/' + str(i) + '.png')
    Image.fromarray(auto_encoded_images_clr1[i]).save('auto_encoded1/' + str(i) + '.png')

########################################################################################################################
