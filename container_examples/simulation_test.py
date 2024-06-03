import sys

output_file_path = sys.argv[1]
parameter = sys.argv[2]
print(f"PLACEHOLDER SIMULATION SCRIPT: Received input {parameter}")

with open(output_file_path, "w") as file:
    file.write(parameter)