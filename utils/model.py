import os
import json
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

tf.random.set_seed(0)


def build_model(layers, act_fn, kernel_init, bias_init, input_shape, output):
    """
    Builds a sequential, fully-connected, model in keras with the given parameters.
    Check the available options in tensorflow for the parameters mentioned below
    Args:
        layers: int
            number of hidden layers (number of hidden units is fixed in 1000 units)
        act_fn: str
            type of activation function
        kernel_init: str
            type of weight initialization
        bias_init: str
            type of bias initialization
        input_shape: tuple
            tuple indicating the input shape of the model
        output: int
            number of units in the output layer of the network
    Returns:
        Sequential model
    """

    model = Sequential([Flatten(input_shape=input_shape)])

    # intermediary fully-connected layers
    for _ in range(layers):
        model.add(
            Dense(
                1000,
                activation=act_fn,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
            )
        )

    # final layer
    model.add(
        Dense(
            output,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            activation="softmax",
        )
    )

    return model


def write_meta(args, path):
    """
    Store metadata from execution
    Args:
        args: dict
            dictionary containing metadata
        path: str
            path or name of the file to store
                If only filename is provided, the file will be saved in "meta/"
    """

    path = "meta/" + path if "/" not in path else path
    path = path.replace(".json", "")

    os.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with open("{}.json".format(path), "w") as f:
        json.dump(args, f)


def save_model(model, path):
    """Save model and current weights
    Args:
        model: tf.keras model
            model to save
        path: str
            path or name of the file to store
                If only filename is provided, the file will be saved in "models/"
    """

    path = "models/" + path if "/" not in path else path
    path = path.replace(".h5", "")

    os.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    model.save("{}.h5".format(path))
    print("\nModel saved successfully on file {}\n".format("{}.h5".format(path)))


def reinitialize_model(meta_path, weight_path=None, return_meta=False):
    """Reinitialize model from metadata
    Args:
        meta_path: str
            path to metadata file
        weight_path: str (default None)
            path to weight file (.h5)
        return_meta: bool (default: False)
            return metadata
    Returns:
        model: tf.keras model
        meta: dict
    """

    with open(meta_path, "r") as f:
        meta = json.load(f)

    model = build_model(
        meta["layers"], meta["act_fn"], meta["kernel"], meta["bias"], meta["input_shape"], meta["outputs"]
    )
    model.compile(optimizer=meta["opt"], loss=meta["loss"], metrics=["accuracy"])

    if weight_path:
        model.load_weights(weight_path)

    if return_meta:
        return model, meta
    else:
        return model


def train(model, meta, data):

    (X_train, y_train), (X_test, y_test) = data
    os.makedirs("./weights/{}/".format(meta["name"]), exist_ok=True)
    writer = tf.summary.create_file_writer("logs/{}/".format(meta["name"]))

    with writer.as_default():
        for step in range(meta["cur_epoch"], meta["epochs"]):
            print("Epoch :{}/{}".format(step, meta["epochs"]))
            log = model.fit(
                X_train,
                y_train,
                verbose=0,
                batch_size=meta["batch_size"],
                validation_data=(X_test, y_test),
            )

            for key, value in log.history.items():
                tf.summary.scalar(key, value[0], step=step)

            if step % meta["save"] == 0:
                build_distribution(model, X_test[:meta["n_examples"]], title="Iteration: {}".format(step), path="./images/{}".format(meta["name"]))
                write_meta(meta["name"], {**meta, **{"cur_epoch": step}})

                model.save_weights("./weights/{}/{}.h5".format(meta["name"], step))
            writer.flush()

    model.save_weights("./weights/{}/{}.h5".format(meta["name"], meta["epochs"]))
    results = model.evaluate(X_test, y_test)
    write_meta(meta["name"], {**meta, **{"evaluation": list(map(str, results))}})