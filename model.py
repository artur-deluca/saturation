import os
import json
import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense

def build_model(layers, act_fn, kernel_init, bias_init, input_shape):

    model = tf.keras.models.Sequential([Flatten(input_shape=input_shape)])

    # intermediary fully-connected layers
    for _ in range(layers):
        model.add(Dense(1000, activation=act_fn, kernel_initializer=kernel_init, bias_initializer=bias_init))

    # final layer
    model.add(Dense(10, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='softmax'))

    return model

def write_meta(name, args):
    if not os.path.exists("meta/"):
        os.makedirs("meta/")

    with open("meta/{}.json".format(name), "w") as f:
        json.dump(args , f)


def save_model(model, name):

    models_dir = './models/'
    filename = os.path.join(models_dir, "{}.h5".format(name))
    model.save(filename)
    print("\nModel saved successfully on file {}\n".format(filename))

		
    