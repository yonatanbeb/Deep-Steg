from PIL import Image
from encode_dataset import encode_clearance_level_0, encode_clearance_level_1
from auto_encoder import auto_encode
import numpy as np


def display_image(img_array):
    display = Image.fromarray(img_array)
    display.show()


def display_pair(image_with_key, clearance_level):
    image_with_key = image_with_key.reshape(28, 28)

    auto_encoded_image = auto_encode(image_with_key, clearance_level)

    trip = np.concatenate([image_with_key, auto_encoded_image], axis=0)
    display_image(trip)


def display_triplet(orig_image, clearance_level):
    image_with_key, _ = encode_clearance_level_0([orig_image], [0]) if clearance_level == 0 \
        else encode_clearance_level_1([orig_image], [0])
    image_with_key = image_with_key.reshape(28, 28)

    auto_encoded_image = auto_encode(orig_image, clearance_level)

    trip = np.concatenate([orig_image, image_with_key, auto_encoded_image], axis=0)
    display_image(trip)
