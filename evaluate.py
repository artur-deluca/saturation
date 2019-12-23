import argparse
import os
import json
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.backend import function

parser = argparse.ArgumentParser(description="Evaluate activation values")
parser.add_argument("--dir", "-d", default="./models", type=str, help="Folder path")
parser.add_argument("--examples", "-e", default=300, type=int, help="Number of examples to evaluate")


if __name__ == "__main__":
    args = parser.parse_args()
    _, (data_x, _)  = cifar10.load_data()
    data_x = data_x[:args.examples]
    for filename in os.listdir(args.dir):

        if filename.endswith(".h5"):

            name = filename.split(".h5")[0]
            path = os.path.join(args.dir, filename)
            model = tf.keras.models.load_model(path)
            
            results = {}
            for i in range(len(model.layers)):
                key = "layer {}".format(i)
                get_act_fn_val = function([model.layers[0].input], [model.layers[i].output])
                results[key] = get_act_fn_val(data_x)[0].flatten().tolist()

        output = os.path.join(args.dir, "act_val")
        if not os.path.exists(output):
            os.makedirs(output)

        with open(os.path.join(output, name) + ".json", "w") as f:
            json.dump(results , f)


        

