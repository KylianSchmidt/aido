import sys
import json

parameter_file_path = sys.argv[1]

# Example of how to access the parameter dict in json format:

with open(parameter_file_path) as file:
    parameter_dict = json.load(file)["Parameters"]

    foo = parameter_dict["foo"]["current_value"]

    print(foo)