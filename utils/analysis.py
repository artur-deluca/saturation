import h5py
import os
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras import Model


def build_distribution(model, X_test, title="", path=None):

    activation = Model(
        model.inputs, [model.layers[i].output for i in range(1, len(model.layers) - 1)]
    )

    values = activation.predict(X_test)
    labels = ["Layer {}".format(i) for i in range(1, len(model.layers) - 1)]

    for i in range(len(labels)):
        sns.kdeplot(values[i].flatten(), shade=True, label=labels[i])

    plt.legend()
    plt.title(title)

    if path:
        os.makedirs(path, exist_ok=True)
        plt.savefig("{}/{}.png".format(path, title))
        plt.clf()
        with h5py.File("{}/values.hdf5".format(path), "a") as f:
            f[title] = [x.flatten() for x in values]
    else:
        plt.show()
