import argparse
import json
from fashion_mnist.fashion_mnist_to_png import images
from auto_encoder import auto_encode
from classifier import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    args = parser.parse_args()

    with open('user_data/current_clearance_level.json') as current_clearance_level_json:
        current_clearance_level = json.load(current_clearance_level_json)

    if current_clearance_level_json == 'No Clearance':
        predict(images[args.index])
    else:
        image = auto_encode(images[args.index], current_clearance_level)
        label = predict(image)

    print(label)


if __name__ == '__main__':
    main()
