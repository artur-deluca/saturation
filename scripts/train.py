import sys
sys.path.insert(0, '')

import argparse
import datetime
import os
import tensorflow as tf

import utils
from autoencoder.auto_encoder import pretrain_model

N_EXAMPLES = 300

print("GPU: {}".format(tf.test.gpu_device_name()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train neural network")
    parser.add_argument("--layers", "-nl", default=5, type=int, help="Number of layers")
    parser.add_argument(
        "--act_fn", "-a", default="sigmoid", type=str, help="Activation function"
    )
    parser.add_argument(
        "--kernel", "-ki", default=None, type=str, help="Kernel initializer"
    )
    parser.add_argument(
        "--bias", "-bi", default="zeros", type=str, help="Bias initializer"
    )
    parser.add_argument(
        "--epochs", "-e", default=140, type=int, help="Number of epochs to train model"
    )
    parser.add_argument(
        "--batch_size", "-b", default=10, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--opt", "-o", default="sgd", type=str, help="Optimizing algorithm"
    )
    parser.add_argument(
        "--loss",
        "-l",
        default="sparse_categorical_crossentropy",
        type=str,
        help="Loss function",
    )
    parser.add_argument(
        "--dataset", "-d", default="cifar10", type=str, help="Dataset"
    )
    parser.add_argument(
        "--save", "-sv", default=10, type=int, help="Save every X iterations"
    )

    parser.add_argument("--pretrain", dest="pretrain", action="store_true")
    parser.add_argument("--no-pretrain", dest="pretrain", action="store_false")
    parser.set_defaults(pretrain=False)

    args = parser.parse_args()

    meta = args.__dict__
    meta["cur_epoch"] = 0
    meta["name"] = args.act_fn + str(args.layers) + args.kernel + str(datetime.datetime.now())
    meta["input_shape"], meta["output"], dataset = utils.get_data(args.dataset)
    

    # build model
    model = utils.build_model(args.layers, args.act_fn, args.kernel, args.bias, meta["input_shape"], meta["output"])
    model.compile(optimizer=args.opt, loss=args.loss, metrics=["accuracy"])

    if args.pretrain:
        model = pretrain_model(model, meta, dataset[0][0][:500], dataset[1][0][:N_EXAMPLES])

    utils.train(model, meta, dataset)
