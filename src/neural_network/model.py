import tensorflow as tf


def build_model(input_shape):
    model = tf.keras.models.Sequential([
        # layers here
    ])

    model.compile(optimizer='...',
                  loss='...',
                  metrics=['...'])

    return model


def save_model(model, path="path/to/save/model"):
    model.save(path)


def load_model(path="path/to/load/from"):
    return tf.keras.models.load_model(path)