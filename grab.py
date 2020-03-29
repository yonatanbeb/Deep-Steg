import os
import argparse
import json
from PIL import Image
from fashion_mnist.fashion_mnist_to_png import images
from neural_nets.auto_encoder import auto_encode
from neural_nets.classifier import predict

label1 = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
label0 = ['Top', 'Bottom', 'Shoe']


def main():
    with open('user_data/current_clearance_level.json') as current_clearance_level_json:
        current_clearance_level = json.load(current_clearance_level_json)

    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                                          'Shirt', 'Sneaker', 'Bag', 'Boot', 'Top', 'Bottom', 'Shoe'])
    parser.add_argument('num', type=int)
    parser.add_argument('to_path', type=str)
    args = parser.parse_args()

    if current_clearance_level == 'No Clearance':
        print('Unauthorized: No Clearance')
    elif current_clearance_level == 0 and args.label in label1:
        print('Requested label unauthorized for User with your clearance level')
    else:
        # if user has clearance level 1, user can also search level 0 labels
        if current_clearance_level == 1 and args.label in label0:
            current_clearance_level = 0
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
