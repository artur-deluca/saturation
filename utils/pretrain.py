import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GaussianNoise
from tensorflow.keras import Sequential


def get_dae(inputs):
    dae = Sequential([inputs])
    dae.add(Dense(inputs.input.shape.as_list()[1]))
    return dae


def pretrain(model, X_train, X_test, epochs, batch_size, opt, name, fn, save):
    writer = tf.summary.create_file_writer("logs/{}/".format(name))
    step = 0
    for i in range(1, len(model.layers)):
        print("Pretraining layer {}".format(i))
        if not isinstance(model.layers[i], Dense):
            print(
                "layer {}:{} is not a Dense layer, skipping".format(
                    i, type(model.layers[i])
                )
            )
            continue

        embedder = Model(
            model.get_layer(index=0).input, model.get_layer(index=i - 1).output
        )
        data = embedder.predict(X_train)

        noise = tf.random.normal(
            shape=data.shape, mean=0.0, stddev=1, dtype=tf.float32
        ).numpy()
        corrupted = data + noise
        dae = get_dae(model.get_layer(index=i))
        dae.compile(loss="mean_squared_error", optimizer=opt)
        for _ in range(epochs):
            log = dae.fit(
                corrupted, data, batch_size=batch_size, verbose=0, validation_split=0.2
            )
            for key, value in log.history.items():
                tf.summary.scalar(key, value[0], step=step)

            if step % 5 == 0:
                model.layers[i].set_weights(dae.layers[0].get_weights())
                model.layers[i].trainable = False
                fn(
                    model,
                    X_test,
                    title="Pretrain: {}".format(step),
                    path="./images/{}".format(name),
                )
            writer.flush()
            step += 1
        model.layers[i].set_weights(dae.layers[0].get_weights())
        model.layers[i].trainable = False

    for layer in model.layers:
        layer.trainable = True

    return model
