import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    View_Encoded = parser.add_argument_group(description="VIEW VERSIONS WITH ENCODING:")
    View_Encoded.add_argument('-v', '--version', choices=['encoded', 'auto_encoded'],
                              help='request stage of encoding: omit for original images')
    group = View_Encoded.add_mutually_exclusive_group(required='--version' in sys.argv)
    group.add_argument('-z', '--zero', action='store_true', help='view images with clearance level 0')
    group.add_argument('-o', '--one', action='store_true', help='view images with clearance level 1')
    args = parser.parse_args()

    if args.version is None:
        os.system('xdg-open /original')
    elif args.zero:
        os.system('xdg-open /' + args.version + '0')
    else:
        os.system('xdg-open /' + args.version + '1')


if __name__ == '__main__':
    main()
