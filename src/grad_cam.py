import PIL
import tensorflow as tf
import numpy as np
import xception_cross_stitched
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Image, display

def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, grad_model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.Model(inputs=grad_model.input, outputs=[grad_model.get_layer(last_conv_layer_name).output, grad_model.output]
     )
   
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        print("preds", preds)

        ##softmax
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        print("Pred index", pred_index)
        print("Class channel", class_channel)
     
    grads = tape.gradient(class_channel, last_conv_layer_output)
    #print("grads", grads)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap.save('/content/cam_2.jpg')
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img = superimposed_img.resize((100, 100))
    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

def main():
    INPUT_SHAPE = (299, 299, 3)
    img = PIL.Image.open("image.png")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    print("image array shape", img_array.shape)
  
    #F2F
    layer = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.08027123, 0.04175776, 0.09654982]), variance=np.array([2.6944017, 2.7303824, 2.649358]))
 
    model = xception_cross_stitched.build_model(normalize=layer)
    learning_rate = 2e-4
    loss = tf.keras.losses.sparse_categorical_crossentropy
    metrics = ["accuracy"]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=metrics)

    load_status = model.load_weights("ckpt.h5")

    last_conv_layer_name = "block14_sepconv2_act"
    # Remove last layer's softmax
    model.layers[-1].activation = None
    print(model.get_layer(name=last_conv_layer_name))
    #model.summary()
    print(len(model.layers))
    preds = model(img_array)
    print("Prediction", preds)

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)


    # Display heatmap
    plt.matshow(heatmap)
    plt.show()
    save_and_display_gradcam('image.png', heatmap)

if __name__ == "__main__":
    main()
