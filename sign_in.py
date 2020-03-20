import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('username', type=str)
    parser.add_argument('password', type=str)

    args = parser.parse_args()

    with open('profiles.json') as profiles_json:
        profiles = json.load(profiles_json)

    if args.username in profiles:
        if profiles[args.username][0] == args.password:
            current_clearance_level = profiles[args.username][1]
        else:
            current_clearance_level = 'No Clearance'
            print('Incorrect Password')
    else:
        current_clearance_level = 'No Clearance'
        print('Incorrect Username')

    current_clearance_level_json = json.dumps(current_clearance_level)
    with open('current_clearance_level.json', 'w') as current_clearance_level:
        current_clearance_level.write(current_clearance_level_json)


if __name__ == '__main__':
    main()
