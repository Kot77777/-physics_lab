import json

config = {
    "x_0": 1,
    "v_0": 0,
    "start_time": 0,
    "dt": 0.1,
    "end_time": 20,
    "w": 1
}

with open('config.json', 'w') as json_file:
    json.dump(config, json_file, indent=4)