import os
import argparse
import json
from PIL import Image
from fashion_mnist.fashion_mnist_to_png import images
from auto_encoder import auto_encode
from classifier import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', type=str)
    parser.add_argument('num', type=int)
    parser.add_argument('to_path', type=str)
    args = parser.parse_args()

    with open('user_data/current_clearance_level.json') as current_clearance_level_json:
        current_clearance_level = json.load(current_clearance_level_json)

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
