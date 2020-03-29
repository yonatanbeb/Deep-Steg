import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('username', type=str)
    parser.add_argument('password', type=str)
    parser.add_argument('clearance_level', type=int, choices=[0, 1])

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
        if args.clearance_level > profiles[args.username][1]:
            print('User requested higher clearance level \nUser ' + args.username + ' has level 0 clearance.')
        elif args.clearance_level < profiles[args.username][1]:
            if input('User has higher clearance: \nDo you wish to lower your clearance level to 0 [Enter Y]?') == 'Y':
                profiles[args.username][1] = args.clearance_level
                profiles_json = json.dumps(profiles)
                with open('user_data/profiles.json', 'w') as profiles:
                    profiles.write(profiles_json)


if __name__ == '__main__':
    main()
