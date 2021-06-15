import numpy as np
import xceptionnet
from models import build_xception
import tensorflow as tf
from tensorflow import keras
import PIL
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def main():
    INPUT_SHAPE = (299, 299, 3)
    img = PIL.Image.open("/workspace/data/classification/c23/c23_xception/all/fake/NeuralTextures/train/350_349_0060.png")
    img_array = tf.keras.preprocessing.image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)
    print("image array shape", img_array.shape)
    base_model = xceptionnet.Xception(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    model = build_xception(base_model, input_shape=INPUT_SHAPE)

    
    learning_rate = 2e-4
    #loss = tf.keras.losses.binary_crossentropy
    #metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.AUC()]
    loss = tf.keras.losses.sparse_categorical_crossentropy
    metrics = ["accuracy"]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

    load_status = model.load_weights("/workspace/data/trained_models/checkpoint/c23/ckpt_finetune_NT_acc.h5")
    model.summary()

    last_conv_layer_name = "block14_sepconv2_act"
    # Remove last layer's softmax
    model.layers[-1].activation = None
    model.summary()
    # Print what the top predicted class is
    preds = model.predict(img_array)
    print("Prediction", preds)

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()


if __name__ == "__main__":
    main()
