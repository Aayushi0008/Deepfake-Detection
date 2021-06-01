import tensorflow as tf
import cross_stitch
from tensorflow.python.keras.utils import data_utils
layers = tf.keras.layers


def build_model(normalize=None):
    input_shape = (299, 299, 3)
    img_input = layers.Input(shape=input_shape)

    channel_axis = -1
    
    img_input_1 = doDWT(img_input)
    x_norm = layers.experimental.preprocessing.Rescaling(scale=1.0 / 127.5, offset=-1)
    img_input_2 = x_norm(img_input_1)
    #img_input_2 = tf.pad(img_input_2, paddings=[[0, 0], [0, 149], [0, 149], [0, 0]])

    x = layers.Conv2D(
        32, (3, 3),
        strides=(2, 2),
        use_bias=False,
        name='block1_conv1_dct')(img_input_2)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn_dct')(x)
    x = layers.Activation('relu', name='block1_conv1_act_dct')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2_dct')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn_dct')(x)
    x = layers.Activation('relu', name='block1_conv2_act_dct')(x)

    residual = layers.Conv2D(
        128, (1, 1), strides=(2, 2), padding='same', use_bias=False, name="residual_1_dct")(x)
    residual = layers.BatchNormalization(axis=channel_axis, name="residual_1_bn_dct")(residual)

    x = layers.SeparableConv2D(
        128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1_dct')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn_dct')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act_dct')(x)
    x = layers.SeparableConv2D(
        128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2_dct')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn_dct')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool_dct')(x)
    x = layers.add([x, residual], name="merge_1_dct")

    residual = layers.Conv2D(
        256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name="residual_2_dct")(x)
    residual = layers.BatchNormalization(axis=channel_axis, name="residual_2_bn_dct")(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act_dct')(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1_dct')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn_dct')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act_dct')(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2_dct')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn_dct')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block3_pool_dct')(x)
    x = layers.add([x, residual], name="merge_2_dct")

    residual = layers.Conv2D(
        728, (1, 1), strides=(2, 2), padding='same', use_bias=False, name="residual_3_dct")(x)
    residual = layers.BatchNormalization(axis=channel_axis, name="residual_3_bn_dct")(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act_dct')(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1_dct')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn_dct')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act_dct')(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2_dct')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn_dct')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block4_pool_dct')(x)
    x = layers.add([x, residual], name="merge_3_dct")

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act_dct')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv1_dct')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv1_bn_dct')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act_dct')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv2_dct')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv2_bn_dct')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act_dct')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv3_dct')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv3_bn_dct')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(
        1024, (1, 1), strides=(2, 2), padding='same', use_bias=False, name="residual_4_dct")(x)
    residual = layers.BatchNormalization(axis=channel_axis, name="residual_4_bn_dct")(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act_dct')(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1_dct')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block13_sepconv1_bn_dct')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act_dct')(x)
    x = layers.SeparableConv2D(
        1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2_dct')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block13_sepconv2_bn_dct')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool_dct')(x)
    x = layers.add([x, residual], name="merge_4_dct")

    x = layers.SeparableConv2D(
        1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1_dct')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block14_sepconv1_bn_dct')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act_dct')(x)

    x = layers.SeparableConv2D(
        2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2_dct')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block14_sepconv2_bn_dct')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act_dct')(x)

    inputs = img_input
    # Create model.
    model_dct = tf.keras.Model(inputs, x, name='xception_dct')

    
    x_norm = layers.experimental.preprocessing.Rescaling(scale=1.0 / 127.5, offset=-1)
    img_input_1 = x_norm(img_input)
    img_input_1 = tf.pad(img_input_1, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])
    max_pool = tf.keras.layers.MaxPooling2D()
    img_input_2 = max_pool(img_input_1)
    x = layers.Conv2D(
        32, (3, 3),
        strides=(2, 2),
        use_bias=False,
        name='block1_conv1')(img_input_2)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(
        128, (1, 1), strides=(2, 2), padding='same', use_bias=False, name="residual_1")(x)
    residual = layers.BatchNormalization(axis=channel_axis, name="residual_1_bn")(residual)

    x = layers.SeparableConv2D(
        128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual], name="merge_1")

    residual = layers.Conv2D(
        256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name="residual_2")(x)
    residual = layers.BatchNormalization(axis=channel_axis, name="residual_2_bn")(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual], name="merge_2")

    residual = layers.Conv2D(
        728, (1, 1), strides=(2, 2), padding='same', use_bias=False, name="residual_3")(x)
    residual = layers.BatchNormalization(axis=channel_axis, name="residual_3_bn")(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual], name="merge_3")

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(
        1024, (1, 1), strides=(2, 2), padding='same', use_bias=False, name="residual_4")(x)
    residual = layers.BatchNormalization(axis=channel_axis, name="residual_4_bn")(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual], name="merge_4")

    x = layers.SeparableConv2D(
        1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(
        2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    inputs = img_input
    # Create model.
    model = tf.keras.Model(inputs, x, name='xception')
    TF_WEIGHTS_PATH_NO_TOP = (
        'https://storage.googleapis.com/tensorflow/keras-applications/'
        'xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
    weights_path = data_utils.get_file(
        'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
        TF_WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='b0042744bf5b25fce3cb969f33bebb97')
    model.load_weights(weights_path)

    # Cross Stitch
    tops = [model.get_layer("merge_1").output, model_dct.get_layer("merge_1_dct").output]
    cs = cross_stitch.CrossStitch(2, name="cross_stitch_1")(tops)
    tops = tf.unstack(cs, axis=0)
    x = tops[0]
    y = tops[1]
    

    residual = model.get_layer("residual_2")(x)
    residual = model.get_layer("residual_2_bn")(residual)
    x = model.get_layer("block3_sepconv1_act")(x)
    x = model.get_layer("block3_sepconv1")(x)
    x = model.get_layer("block3_sepconv1_bn")(x)
    x = model.get_layer("block3_sepconv2_act")(x)
    x = model.get_layer('block3_sepconv2')(x)
    x = model.get_layer('block3_sepconv2_bn')(x)
    x = model.get_layer('block3_pool')(x)
    x = layers.add([x, residual])

    residual = model_dct.get_layer("residual_2_dct")(y)
    residual = model_dct.get_layer("residual_2_bn_dct")(residual)
    y = model_dct.get_layer("block3_sepconv1_act_dct")(y)
    y = model_dct.get_layer("block3_sepconv1_dct")(y)
    y = model_dct.get_layer("block3_sepconv1_bn_dct")(y)
    y = model_dct.get_layer("block3_sepconv2_act_dct")(y)
    y = model_dct.get_layer('block3_sepconv2_dct')(y)
    y = model_dct.get_layer('block3_sepconv2_bn_dct')(y)
    y = model_dct.get_layer('block3_pool_dct')(y)
    y = layers.add([y, residual])

    # Cross Stitch
    tops = [x, y]
    cs = cross_stitch.CrossStitch(2, name="cross_stitch_2")(tops)
    tops = tf.unstack(cs, axis=0)
    x = tops[0]
    y = tops[1]

    residual = model.get_layer("residual_3")(x)
    residual = model.get_layer("residual_3_bn")(residual)
    x = model.get_layer('block4_sepconv1_act')(x)
    x = model.get_layer('block4_sepconv1')(x)
    x = model.get_layer('block4_sepconv1_bn')(x)
    x = model.get_layer('block4_sepconv2_act')(x)
    x = model.get_layer('block4_sepconv2')(x)
    x = model.get_layer('block4_sepconv2_bn')(x)
    x = model.get_layer('block4_pool')(x)
    x = layers.add([x, residual])

    residual = model_dct.get_layer("residual_3_dct")(y)
    residual = model_dct.get_layer("residual_3_bn_dct")(residual)
    y = model_dct.get_layer('block4_sepconv1_act_dct')(y)
    y = model_dct.get_layer('block4_sepconv1_dct')(y)
    y = model_dct.get_layer('block4_sepconv1_bn_dct')(y)
    y = model_dct.get_layer('block4_sepconv2_act_dct')(y)
    y = model_dct.get_layer('block4_sepconv2_dct')(y)
    y = model_dct.get_layer('block4_sepconv2_bn_dct')(y)
    y = model_dct.get_layer('block4_pool_dct')(y)
    y = layers.add([y, residual])

    # Cross Stitch
    tops = [x, y]
    cs = cross_stitch.CrossStitch(2, name="cross_stitch_3")(tops)
    tops = tf.unstack(cs, axis=0)
    x = tops[0]
    y = tops[1]

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = model.get_layer(prefix + '_sepconv1_act')(x)
        x = model.get_layer(prefix + '_sepconv1')(x)
        x = model.get_layer(prefix + '_sepconv1_bn')(x)
        x = model.get_layer(prefix + '_sepconv2_act')(x)
        x = model.get_layer(prefix + '_sepconv2')(x)
        x = model.get_layer(prefix + '_sepconv2_bn')(x)
        x = model.get_layer(prefix + '_sepconv3_act')(x)
        x = model.get_layer(prefix + '_sepconv3')(x)
        x = model.get_layer(prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = model.get_layer("residual_4")(x)
    residual = model.get_layer("residual_4_bn")(residual)
    x = model.get_layer('block13_sepconv1_act')(x)
    x = model.get_layer('block13_sepconv1')(x)
    x = model.get_layer('block13_sepconv1_bn')(x)
    x = model.get_layer('block13_sepconv2_act')(x)
    x = model.get_layer('block13_sepconv2')(x)
    x = model.get_layer('block13_sepconv2_bn')(x)
    x = model.get_layer('block13_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = y
        prefix = 'block' + str(i + 5)

        y = model_dct.get_layer(prefix + '_sepconv1_act_dct')(y)
        y = model_dct.get_layer(prefix + '_sepconv1_dct')(y)
        y = model_dct.get_layer(prefix + '_sepconv1_bn_dct')(y)
        y = model_dct.get_layer(prefix + '_sepconv2_act_dct')(y)
        y = model_dct.get_layer(prefix + '_sepconv2_dct')(y)
        y = model_dct.get_layer(prefix + '_sepconv2_bn_dct')(y)
        y = model_dct.get_layer(prefix + '_sepconv3_act_dct')(y)
        y = model_dct.get_layer(prefix + '_sepconv3_dct')(y)
        y = model_dct.get_layer(prefix + '_sepconv3_bn_dct')(y)
        y = layers.add([y, residual])

    residual = model_dct.get_layer("residual_4_dct")(y)
    residual = model_dct.get_layer("residual_4_bn_dct")(residual)
    y = model_dct.get_layer('block13_sepconv1_act_dct')(y)
    y = model_dct.get_layer('block13_sepconv1_dct')(y)
    y = model_dct.get_layer('block13_sepconv1_bn_dct')(y)
    y = model_dct.get_layer('block13_sepconv2_act_dct')(y)
    y = model_dct.get_layer('block13_sepconv2_dct')(y)
    y = model_dct.get_layer('block13_sepconv2_bn_dct')(y)
    y = model_dct.get_layer('block13_pool_dct')(y)
    y = layers.add([y, residual])

    # Cross Stitch
    tops = [x, y]
    cs = cross_stitch.CrossStitch(2, name="cross_stitch_4")(tops)
    tops = tf.unstack(cs, axis=0)
    x = tops[0]
    y = tops[1]

    x = model.get_layer('block14_sepconv1')(x)
    x = model.get_layer('block14_sepconv1_bn')(x)
    x = model.get_layer('block14_sepconv1_act')(x)
    x = model.get_layer('block14_sepconv2')(x)
    x = model.get_layer('block14_sepconv2_bn')(x)
    x = model.get_layer('block14_sepconv2_act')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)

    y = model_dct.get_layer('block14_sepconv1_dct')(y)
    y = model_dct.get_layer('block14_sepconv1_bn_dct')(y)
    y = model_dct.get_layer('block14_sepconv1_act_dct')(y)
    y = model_dct.get_layer('block14_sepconv2_dct')(y)
    y = model_dct.get_layer('block14_sepconv2_bn_dct')(y)
    y = model_dct.get_layer('block14_sepconv2_act_dct')(y)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dropout(0.5)(y)
    y = layers.Dense(256, activation='relu')(y)

    z = tf.keras.layers.concatenate([x, y], axis=1)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1, activation="sigmoid")(z)
    
    new_model = tf.keras.Model(model.layers[0].input, z)
    print(len(new_model.layers))
    return new_model
    

def doDCT(input):
    input = tf.transpose(input, perm=[0, 3, 1, 2])
    input = tf.signal.dct(input, type=2, axis=-1, norm="ortho")
    input = tf.transpose(input, perm=[0, 1, 3, 2])
    input = tf.signal.dct(input, type=2, axis=-1, norm="ortho")
    input = tf.transpose(input, perm=[0, 3, 2, 1])

    return input


def doInvDCT(input):
    input = tf.transpose(input, perm=[0, 3, 1, 2])
    input = tf.signal.idct(input, type=2, n=None, axis=-1, norm="ortho", name=None)
    input = tf.transpose(input, perm=[0, 1, 3, 2])
    input = tf.signal.idct(input, type=2, n=None, axis=-1, norm="ortho", name=None)
    input = tf.transpose(input, perm=[0, 3, 2, 1])

    return input


def log10(x):
    epsilon = 1e-12
    x = tf.math.abs(x) + epsilon
    num = tf.math.log(x)
    den = tf.math.log(tf.constant(10, dtype=num.dtype))
    return num / den


def log(x):
    epsilon = 1e-12
    x = tf.math.abs(x) + epsilon
    return tf.math.log(x)


def doDWT(x):
    # (batch_size, num_channels, height, width)
    # Change to (batch_size, height, width, num_channels)
    if x.shape[1] % 2 != 0:
        x = tf.pad(x, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]], mode='SYMMETRIC')

    x01 = x[:, 0::2, :, :] / 2
    # x02 = x[:, :, 1::2, :] / 2
    x02 = x[:, 1::2, :, :] / 2
    # x1 = x01[:, :, :, 0::2]
    x1 = x01[:, :, 0::2, :]
    # x2 = x02[:, :, :, 0::2]
    x2 = x02[:, :, 0::2, :]
    # x3 = x01[:, :, :, 1::2]
    x3 = x01[:, :, 1::2, :]
    # x4 = x02[:, :, :, 1::2]
    x4 = x02[:, :, 1::2, :]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return tf.concat((x_LL, x_HL, x_LH, x_HH), 3)
