import argparse
import csv
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

from model import build_model, write_meta, save_model


INPUT_SHAPE = (32, 32, 3)

print("GPU: {}".format(tf.test.gpu_device_name()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train neural network")
    parser.add_argument("--layers", "-nl", default=5, type=int, help="Number of layers")
    parser.add_argument("--act_fn", "-a", default="sigmoid", type=str, help="Activation function")
    parser.add_argument("--kernel", "-ki", default=None, type=str, help="Kernel initializer")
    parser.add_argument("--bias", "-bi", default=None, type=str, help="Bias initializer")
    parser.add_argument("--epochs", "-e", default=140, type=int, help="Number of epochs to train model")
    parser.add_argument("--batch_size", "-b", default=10, type=int, help="Batch size for training")
    parser.add_argument("--opt", "-o", default="sgd", type=str, help="Optimizing algorithm")
    parser.add_argument("--loss", "-l", default="sparse_categorical_crossentropy", type=str, help="Loss function")
    parser.add_argument("--val", "-v", default=.3, type=float, help="Validation set size")
    parser.add_argument("--seed", "-s", default=42, type=float, help="Random seed generator")

    args = parser.parse_args()
    name = args.act_fn + str(args.layers) + args.kernel + str(datetime.datetime.now())

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train/255
    X_test = X_test/255

    # build model
    model = build_model(
        args.layers,
        args.act_fn,
        args.kernel,
        args.bias,
        INPUT_SHAPE
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/{}/".format(name),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]

    model.compile(optimizer=args.opt, loss=args.loss, metrics=["accuracy"])

    model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_test, y_test), callbacks=callbacks, use_multiprocessing=True)

    save_model(model, name)

    results = model.evaluate(X_test, y_test, use_multiprocessing=True)
    write_meta(name, {**args.__dict__, **{"evaluation": list(map(str, results))}})
