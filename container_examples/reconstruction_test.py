import sys

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
print(f"PLACEHOLDER RECONSTRUCTION SCRIPT: Received input file '{output_file_path}'")

with open(input_file_path, "r") as input_file:
    with open(output_file_path, "w") as output_file:
        for line in input_file.readlines():
            output_file.write(line+"\n")
