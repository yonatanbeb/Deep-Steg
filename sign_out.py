import json


def main():
    current_clearance_level = 'No Clearance'
    current_clearance_level_json = json.dumps(current_clearance_level)
    with open('current_clearance_level.json', 'w') as current_clearance_level:
        current_clearance_level.write(current_clearance_level_json)


if __name__ == '__main__':
    main()
