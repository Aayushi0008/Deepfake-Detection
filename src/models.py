import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_shallow_cnn(input_shape, classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(3, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = layers.AveragePooling2D()(x)  # 64

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.AveragePooling2D()(x)  # 32

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = layers.Dense(classes, activation=activation)(x)

    model = keras.Model(inputs, outputs)
    return model


def build_xception(model, input_shape):
    inputs = keras.Input(shape=input_shape)
    x = model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation='relu', name="fc_layer")(x)
    outputs = keras.layers.Dense(2, activation='softmax')(x)
    #outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model


def doDCT(input):
    input = tf.transpose(input, perm=[0, 3, 1, 2])
    input = tf.signal.dct(input, type=2, n=None, axis=-1, norm="ortho", name=None)
    input = tf.transpose(input, perm=[0, 1, 3, 2])
    input = tf.signal.dct(input, type=2, n=None, axis=-1, norm="ortho", name=None)
    input = tf.transpose(input, perm=[0, 3, 2, 1])

    return input


def doInvDCT(input):
    input = tf.transpose(input, perm=[0, 3, 1, 2])
    input = tf.signal.idct(input, type=2, n=None, axis=-1, norm="ortho", name=None)
    input = tf.transpose(input, perm=[0, 1, 3, 2])
    input = tf.signal.idct(input, type=2, n=None, axis=-1, norm="ortho", name=None)
    input = tf.transpose(input, perm=[0, 3, 2, 1])

    return input
