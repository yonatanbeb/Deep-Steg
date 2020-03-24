from keras.datasets import fashion_mnist
import sys
sys.path.append('../')
from dataset_tools.encode_dataset import encode_image_with_0, encode_image_with_1
from neural_nets.auto_encoder import auto_encode
from PIL import Image

(images, _), (_, _) = fashion_mnist.load_data()
images = images[:100]

# for i in range(100):
#     image = Image.fromarray(images[i])
#     image.save('original/' + str(i) + '.png')
#     encoded_image_0 = Image.fromarray(encode_image_with_0(images[i]))
#     encoded_image_0.save('encoded0/' + str(i) + '.png')
#     encoded_image_1 = Image.fromarray(encode_image_with_1(images[i]))
#     encoded_image_1.save('encoded1/' + str(i) + '.png')
#     auto_encoded_image_0 = Image.fromarray(auto_encode(images[i], 0))
#     auto_encoded_image_0.save('auto_encoded0/' + str(i) + '.png')
#     auto_encoded_image_1 = Image.fromarray(auto_encode(images[i], 1))
#     auto_encoded_image_1.save('auto_encoded1/' + str(i) + '.png')
