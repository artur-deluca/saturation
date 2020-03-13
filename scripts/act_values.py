import sys

sys.path.insert(0, "")

import utils

META_PATH = ""
WEIGHT_PATH = ""
N_EXAMPLES = 300

if __name__ == "__main__":

    model, meta = utils.reinitialize_model(META_PATH, WEIGHT_PATH, return_meta=True)
    _, _, _, (X_test, y_test) = utils.get_data(meta["dataset"])

    utils.build_distribution(model, X_test[:N_EXAMPLES])
