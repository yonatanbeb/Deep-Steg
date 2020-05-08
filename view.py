""" command line command to view the different stages of fashion_mnist steganography """
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    # arguments that specify a certain stage of steganography
    View_Encoded = parser.add_argument_group(description="VIEW VERSIONS WITH ENCODING:")
    View_Encoded.add_argument('-v', '--version', choices=['encoded', 'auto_encoded'],
                              help='request stage of encoding: omit for original images')
    group = View_Encoded.add_mutually_exclusive_group(required='--version' in sys.argv)
    group.add_argument('-z', '--zero', action='store_true', help='view images with clearance level 0')
    group.add_argument('-o', '--one', action='store_true', help='view images with clearance level 1')
    args = parser.parse_args()

    # default (no arguments) opens the original fashion_mnist dataset
    if args.version is None:
        os.system('xdg-open datasets/original')
    # if argument --zero selected
    elif args.zero:
        # open requested version (steganography stage) with clearance 0
        os.system('xdg-open datasets/' + args.version + '0')
    # if argument --one selected
    else:
        # open requested version (steganography stage) with clearance 1
        os.system('xdg-open datasets/' + args.version + '1')


if __name__ == '__main__':
    main()
