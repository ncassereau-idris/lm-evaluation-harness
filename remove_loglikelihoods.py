import argparse
import jsonlines

args = argparse.ArgumentParser()
args.add_argument('-i', '--input', help='Input file', required=True)
args.add_argument('-o', '--output', help='Output file', required=True)
args = args.parse_args()

if __name__ == "__main__":
    with jsonlines.open(args.input, 'r') as reader, jsonlines.open(args.output, 'w') as writer:
        for obj in reader:
            del obj['loglikelihoods']
            writer.write(obj)
