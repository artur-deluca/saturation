import sys

sys.path.insert(0, "")

import argparse
import utils

# Just place the path of folder below
default = ""
parser = argparse.ArgumentParser(description="Create animation with the plots")
parser.add_argument("path", default=default, type=str, help="path to folder", nargs="?")
args = parser.parse_args()


if __name__ == "__main__":
    utils.create_gif(args.path)
