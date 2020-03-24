import os
import argparse
import json
from PIL import Image
from fashion_mnist.fashion_mnist_to_png import images
from neural_nets.auto_encoder import auto_encode
from neural_nets.classifier import predict


def main():
    with open('user_data/current_clearance_level.json') as current_clearance_level_json:
        current_clearance_level = json.load(current_clearance_level_json)

    parser = argparse.ArgumentParser()
    if current_clearance_level == 1:
        parser.add_argument('label', choices=['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                                              'Shirt', 'Sneaker', 'Bag', 'Boot'])
    else:
        parser.add_argument('label', choices=['Top', 'Bottom', 'Shoe'])
    parser.add_argument('num', default=1, type=int)
    parser.add_argument('to_path', type=str)
    args = parser.parse_args()

    if current_clearance_level == 'No Clearance':
        print('Unauthorized: No Clearance')
    else:
        os.system('mkdir ' + args.to_path)
        num = 0
        for i in range(len(images)):
            if num == args.num:
                break
            image = auto_encode(images[i], current_clearance_level)
            label = predict(image)
            if label == args.label:
                num += 1
                image = Image.fromarray(images[i])
                image.save('./' + args.to_path + '/' + str(i) + '.png')
        os.system('xdg-open ' + args.to_path)


if __name__ == '__main__':
    main()
