import json
import os


def main():
    current_clearance_level = 'No Clearance'
    current_clearance_level_json = json.dumps(current_clearance_level)
    with open('user_data/current_clearance_level.json', 'w') as current_clearance_level:
        current_clearance_level.write(current_clearance_level_json)

    os.system('echo "$(tput setaf 196)mnist&Co@ : $(tput setaf 178)" > ./user_data/prompt')


if __name__ == '__main__':
    main()
