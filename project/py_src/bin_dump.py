import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to the file to dump')
args = parser.parse_args()

with open(args.path, 'rb') as f:
    data = f.read()
    for byte in data:
        bin_str = bin(byte)[2:]
        bin_str = '0' * (8 - len(bin_str)) + bin_str
        print(bin_str, end='')

