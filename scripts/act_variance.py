import sys

sys.path.insert(0, "")

import argparse
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import natural_sort as sort

PATH = "./data/imgs/tanh_lecun.hdf5"
TITLE = ""

parser = argparse.ArgumentParser(description="Variance plot of the activation values")
parser.add_argument(
    "path",
    default=PATH,
    type=str,
    help="path to activation values .hdf5 file",
    nargs="?",
)
parser.add_argument(
    "title", default=TITLE, type=str, help="title of the image", nargs="?"
)
args = parser.parse_args()

if __name__ == "__main__":
    f = h5py.File(args.path, "r")
    variance = list()
    keys = sort([x for x in f.keys() if "Pretrain" in x])
    pretrain = len(keys)
    keys += sort([x for x in f.keys() if "Iteration" in x])

    for i in keys:
        variance.append(np.var(f[i], axis=1))
    variance = np.array(variance).T

    for i in range(len(variance)):
        ax = sns.lineplot(
            x=range(len(variance[0])), y=variance[i], label="Layer {}".format(i + 1)
        )

    x_pos = (pretrain - min(ax.get_xlim())) / (max(ax.get_xlim()) - min(ax.get_xlim()))

    if pretrain:
        ax.annotate(
            "End pretraining",
            xy=(x_pos, 0),
            xycoords="axes fraction",
            xytext=(x_pos, 0.1),
            textcoords="axes fraction",
            arrowprops=dict(facecolor="grey", shrink=0.1),
            horizontalalignment="right",
            verticalalignment="top",
        )

    plt.title(args.title)
    plt.savefig(args.path.replace(".hdf5", ".svg"))

