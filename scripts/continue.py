import sys
sys.path.insert(0, '')

import os
import tensorflow as tf

import utils

tf.random.set_seed(0)

META_PATH = "../meta/sigmoid5RandomUniform2020-03-07 20:06:15.102514.json"
WEIGHT_PATH = "../weights/140.h5"

name = META_PATH.rsplit("/", 1)[-1].strip(".json")

print("GPU: {}".format(tf.test.gpu_device_name()))

if __name__ == "__main__":

    model, meta = utils.reinitialize_model(META_PATH, WEIGHT_PATH, return_meta=True)
    _, _, dataset = utils.get_data(meta["dataset"])
    utils.train(model, {**meta, **{"name": name}}, dataset)