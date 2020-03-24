import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('username', type=str)
    parser.add_argument('password', type=str)
    parser.add_argument('clearance_level', choices=[0, 1])

    with open('user_data/profiles.json') as profiles_json:
        profiles = json.load(profiles_json)

    args = parser.parse_args()

    if args.username not in profiles:
        profiles[args.username] = [args.password, args.clearance_level]
        profiles_json = json.dumps(profiles)
        with open('user_data/profiles.json', 'w') as profiles:
            profiles.write(profiles_json)
    else:
        print('Profile Exists')

    os.system('python sign_in.py ' + args.username + ' ' + args.password)


if __name__ == '__main__':
    main()
