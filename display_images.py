from keras.models import load_model
from PIL import Image
from auto_encode_dataset import x_train
from mnist import zeros, ones
from random import randrange
import numpy as np

len0 = len(zeros)
len1 = len(ones)


def display_image(img_array):
    display = Image.fromarray(img_array)
    display.show()


def display_triplet_clr0(index):
    image = x_train[index]

    Encoder = load_model('./models/encoder_0.h5')
    Decoder = load_model('./models/decoder_0.h5')

    image_with_key = image + zeros[randrange(len0)]
    a = (image_with_key.astype('float32') / 255).reshape(1, np.prod(image_with_key.shape))

    image_encoding = Encoder.predict(a)
    encoded_image = (Decoder.predict(image_encoding).astype('uint8') * 255).reshape(28, 28)

    trip = np.concatenate([image, image_with_key, encoded_image], axis=0)
    display_image(trip)


def display_triplet_clr1(index):
    image = x_train[index]
    Encoder = load_model('./models/encoder_1.h5')
    Decoder = load_model('./models/decoder_1.h5')

    image_with_key = image + ones[randrange(len1)]
    a = (image_with_key.astype('float32') / 255).reshape(1, np.prod(image_with_key.shape))

    image_encoding = Encoder.predict(a)
    encoded_image = (Decoder.predict(image_encoding).astype('uint8') * 255).reshape(28, 28)
    display_image(encoded_image)
    trip = np.concatenate([image, image_with_key, encoded_image], axis=0)
    display_image(trip)

