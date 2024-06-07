import sys

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(input_file_path, "r") as input_file:
    with open(output_file_path, "w") as output_file:
        output_file.write("1, 2\n 3.0, 4")
