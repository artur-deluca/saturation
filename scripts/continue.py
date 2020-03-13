import sys

sys.path.insert(0, "")

import argparse
import os
import tensorflow as tf

import utils

tf.random.set_seed(0)

META_PATH = ""
WEIGHT_PATH = ""
parser = argparse.ArgumentParser(description="Continue training from metadata file")
parser.add_argument(
    "meta", default=META_PATH, type=str, help="path to metadata file", nargs="?"
)
parser.add_argument(
    "weights", default=WEIGHT_PATH, type=str, help="path to weights", nargs="?"
)
args = parser.parse_args()


name = META_PATH.rsplit("/", 1)[-1].strip(".json")

print("GPU: {}".format(tf.test.gpu_device_name()))

if __name__ == "__main__":

    model, meta = utils.reinitialize_model(args.meta, args.weights, return_meta=True)
    _, _, *dataset = utils.get_data(meta["dataset"])
    utils.train(model, {**meta, **{"name": name}}, dataset)
