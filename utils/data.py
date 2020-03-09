from tensorflow.keras.datasets import cifar10, mnist

def get_data(dataset):
    """
    Fetch and prepare dataset
    Args:
        dataset: str
            available options: "cifar10", "mnist"
    Returns:
        input_shape: tuple
            Shape of the X_train for building the model
        output: int
            Number of classes on y_train
        training_set: ndarray tuples
            (X_train, y_train)
        test_set: ndarray tuples
            (X_test, y_test)
    """
    if dataset.lower() == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train / 255
        X_test = X_test / 255
        input_shape = (32, 32, 3)
    
    elif dataset.lower() == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        input_shape = (28, 28)
    
    else:
        print("dataset not found!")
        return

    output = len(set(y_test.flatten()))
    return input_shape, output, (X_train, y_train), (X_test, y_test)