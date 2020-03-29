from keras.datasets import fashion_mnist
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import sys
sys.path.append('../')
from encode_dataset import encode_0, encode_1

MODEL_PATH = os.path.abspath('../models')

(images, _), (_, _) = fashion_mnist.load_data()
images = images[:100]

encoded_images_0, _ = encode_0(images, (np.ones(len(images))))
encoded_images_1, _ = encode_1(images, (np.ones(len(images))))


def auto_encode(encoded_images, clearance_level):
    Encoder = load_model(MODEL_PATH + '/encoder_' + str(clearance_level) + '.h5')
    Decoder = load_model(MODEL_PATH + '/decoder_' + str(clearance_level) + '.h5')

    encoded_images = (encoded_images.astype('float32') / 255).reshape(encoded_images.shape[0], 784)

    encoded_images = Encoder.predict(encoded_images)
    auto_encoded_images = Decoder.predict(encoded_images)

    auto_encoded_images = (auto_encoded_images * 255).astype('uint8').reshape(auto_encoded_images.shape[0], 28, 28)

    return auto_encoded_images


auto_encoded_images_0 = auto_encode(encoded_images_0, 0)
auto_encoded_images_1 = auto_encode(encoded_images_1, 1)

for i in range(100):
    Image.fromarray(encoded_images_0[i]).save('encoded0/' + str(i) + '.png')
    Image.fromarray(encoded_images_1[i]).save('encoded1/' + str(i) + '.png')
    Image.fromarray(auto_encoded_images_0[i]).save('auto_encoded0/' + str(i) + '.png')
    Image.fromarray(auto_encoded_images_1[i]).save('auto_encoded1/' + str(i) + '.png')

