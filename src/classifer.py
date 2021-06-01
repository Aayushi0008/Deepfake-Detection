import argparse
import datetime as dt
import os

import tensorflow as tf
import numpy as np
import random
np.random.seed(1337)
random.seed(1337)
tf.random.set_seed(1)

import tensorflow.keras.backend as K
import xceptionnet
import xceptionnet_dct
import xception_fused_bin_dwt
import xception_fused_bin
import xception_fused_bin_2
import xception_fused_bin_fft
import xceptionnet_dwt
from dataset import deserialize_data, deserialize_data_
from models import (build_multinomial_regression,
                    build_multinomial_regression_l1,
                    build_multinomial_regression_l1_l2,
                    build_multinomial_regression_l2,
                    build_simple_cnn, build_simple_nn, build_xception, build_resnet)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# complete size
TRAIN_SIZE = 27935
VAL_SIZE = 69901
TEST_SIZE = 100

CLASSES = 2
CHANNEL_DIM = 3
INPUT_SHAPE = [299, 299, 3]

def load_tfrecord(path, train=True, unbounded=True):
    """Load tfrecords."""
    raw_image_dataset = tf.data.TFRecordDataset(path)
    dataset = raw_image_dataset.map(lambda x: deserialize_data(
        x, shape=INPUT_SHAPE), num_parallel_calls=AUTOTUNE)
    if train:
        dataset = dataset.take(TRAIN_SIZE)

    dataset = dataset.batch(BATCH_SIZE)

    if unbounded:
        dataset = dataset.repeat()

    return dataset.prefetch(AUTOTUNE)


def build_model(args):
    input_shape = INPUT_SHAPE
    print("Input Shape", INPUT_SHAPE)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    learning_rate = 2e-4

    # select model
    with mirrored_strategy.scope():
        if args.MODEL == "resnet":
            model = ResNet50(include_top=True, weights=None, input_shape=(224, 224, 3), classes=2,
                                         classifier_activation='softmax')
        elif args.MODEL == "nn":
            model = build_simple_nn(input_shape, CLASSES, l2=args.l2)
        elif args.MODEL == "cnn":
            model = build_simple_cnn(input_shape, CLASSES)
        elif args.MODEL == "log":
            model = build_multinomial_regression(
                input_shape, CLASSES)
        elif args.MODEL == "log1":
            model = build_multinomial_regression_l1(
                input_shape, CLASSES, l_1=args.l1)
        elif args.MODEL == "log2":
            model = build_multinomial_regression_l2(
                input_shape, CLASSES, l_2=args.l2)
        elif args.MODEL == "xceptionnet":
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization()
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.15941712, -0.21857856, -0.14103174]),
            #                                                             variance=np.array([2.650704 , 2.686771 , 2.5989208]))
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.0383478, -0.08740273, -0.0162791]), variance=np.array([2.7453117, 2.7948494, 2.6936328]))
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.12483676, 0.08915708, 0.13885617]),
            #                                                             variance=np.array([2.6640997, 2.6978483, 2.6216369]))
            
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.1031178 , -0.15620254, -0.08823884]),
            #                                                             variance=np.array([2.643245 , 2.6748462, 2.5946023])) 
            
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.12216299, -0.17707355, -0.10506448]),
            #                                                             variance=np.array([2.6579645, 2.6919966, 2.6076467]))

            normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.03693341, -0.00503048,  0.05477703]),
                                                                         variance=np.array([2.7157836, 2.7571144, 2.668294]))
            
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.00458801, -0.05002853,  0.01623038]),
            #                                                             variance=np.array([2.736655 , 2.7816448, 2.6854408]))
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.14348643, -0.19796225, -0.12978919]),
            #                                                             variance=np.array([2.6607969, 2.693048 , 2.6120377]))

           
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.12216299, -0.17707355, -0.10506448]),
            #                                                             variance=np.array([2.6579645, 2.6919966, 2.6076467]))
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.14760466, -0.20490389, -0.13061523]), variance=np.array([2.6515226, 2.686187 , 2.6000836]))
            
            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([2178.1428, 2030.9546, 2020.1321]), variance=np.array([2.0308804e+09, 1.1955764e+09, 1.0291866e+09]))
            #normalize.adapt(getTrainData(args))
            #print("mean", normalize.mean)
            #print("variance", normalize.variance)

            #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.08027123, 0.04175776, 0.09654982]),
            #                                                             variance=np.array([2.6944017, 2.7303824, 2.649358]))
            
            print("mean", normalize.mean_val)
            print("variance", normalize.variance_val)
            model = xception_fused_bin.build_model(normalize=normalize)
            #model = xceptionnet.Xception(include_top=False, input_shape=INPUT_SHAPE)
            #x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
            #x = tf.keras.layers.Dropout(0.2)(x)
            #x = tf.keras.layers.Dense(512, activation='relu', name="fc_layer")(x)
            #outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
            #new_model = tf.keras.Model(inputs=model.input, outputs=outputs)
            #model = new_model
        elif args.MODEL == "xception_spatial":
            model = tf.keras.applications.Xception(include_top=True, input_shape=INPUT_SHAPE, weights=None, classes=2)
        elif args.MODEL == "log3":
            model = build_multinomial_regression_l1_l2(
                input_shape, CLASSES, l_1=args.l1, l_2=args.l2)
        elif args.MODEL == "densenet":
            model = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3), classes=2)
            inputs = keras.Input(shape=(224, 224, 3))
            x = model(inputs)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dropout(0.5)(x)
            outputs = keras.layers.Dense(2, activation='softmax')(x)
            model = keras.Model(inputs, outputs)
        else:
            raise NotImplementedError(
                "Error model you selected not available!")

        
        loss = tf.keras.losses.binary_crossentropy
        #loss = tf.keras.losses.sparse_categorical_crossentropy
        metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy(threshold=0.5), "accuracy"]
        #metrics = ["accuracy"]
        lr_multipliers = {5:["cross_stitch_1", "cross_stitch_2", "cross_stitch_3", "cross_stitch_4"]}
        #lr_multipliers = {5:["_dct"]}
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, dict_lr_schedule=None)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
    model_name = f"{args.MODEL}_{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_batch_{args.batch_size}_learning_rate_{learning_rate}"
    return model, model_name


def getTrainData(args):
    raw_image_dataset = tf.data.TFRecordDataset("/workspace/data/classification/c23/c23_xception/tfrecords/all_color_fft_train_tf/data.tfrecords")
    dataset = raw_image_dataset.map(lambda x: deserialize_data_(
        x, shape=INPUT_SHAPE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset

def scheduler(epoch, lr):
    if epoch < 1:
       return lr
    else:
       return lr * tf.math.exp(-0.1)

def train(args):
    train_dataset = load_tfrecord(args.TRAIN_DATASET)
    val_dataset = load_tfrecord(args.VAL_DATASET)
    model, model_name = build_model(args)
    model_dir = args.MODEL_DIR 
    filepath = "/workspace/data/trained_models/checkpoint/grad_cam/base/NT/ckpt-{epoch:02d}.h5"
    #filepath = "/workspace/data/trained_models/checkpoint/all_c23_base/softmax/ckpt_DF_2.h5"
    if args.debug:
        callbacks = None
    else:
        callbacks = [
            #    tf.keras.callbacks.EarlyStopping(
            #    monitor='val_loss',
            #    patience=50,
            #    restore_best_weights=True,
            #),
             #tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=2,
                factor=0.2,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=False)
            #tf.keras.callbacks.ModelCheckpoint(
            #    filepath="/workspace/data/trained_models/checkpoint/ckpt_loss_NT.h5",
            #    save_weights_only=True,
            #    monitor='val_loss',
            #    mode='min',
            #    save_best_only=True)
        ]

    model.summary()

    model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
              validation_data=val_dataset,
              validation_steps=VAL_SIZE // BATCH_SIZE,
              callbacks=callbacks)
    _, eval_accuracy = model.evaluate(
            val_dataset, steps=VAL_SIZE // BATCH_SIZE, verbose=0)
 
    return model, eval_accuracy, model_dir


def train_and_save_model(args):
    model, eval_accuracy, model_dir = train(args)


def load_weights_pretrained(args):
    test_dataset = load_tfrecord(args.TEST_DATASET, train=False)
    base_model = xceptionnet.Xception(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE,
                                      classifier_activation='softmax')

    inputs = tf.keras.Input(INPUT_SHAPE)
    model = build_xception(base_model, input_shape=INPUT_SHAPE)

    learning_rate = 1e-5
    loss = tf.keras.losses.binary_crossentropy
    metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    load_status = model.load_weights("/workspace/data/trained_models/checkpoint/all_c40_base/ckpt.h5")
    #model.summary()
    model.evaluate(test_dataset, steps=TEST_SIZE // BATCH_SIZE)


def load_weights(args):
    test_dataset = load_tfrecord(args.TEST_DATASET, train=False)
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization()
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.0383478 , -0.08740273, -0.0162791]),
    #                                                                     variance=np.array([2.7453117, 2.7948494, 2.6936328]))
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.08027123, 0.04175776, 0.09654982]),
    #                                                                     variance=np.array([2.6944017, 2.7303824, 2.649358]))
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.1031178 , -0.15620254, -0.08823884]),
    #                                                                     variance=np.array([2.643245 , 2.6748462, 2.5946023])) 
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.12216299, -0.17707355, -0.10506448]),
    #                                                                     variance=np.array([2.6579645, 2.6919966, 2.6076467]))
 
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-1.00458801, -0.05002853,  0.01623038]),
    #                                                                     variance=np.array([2.736655 , 2.7816448, 2.6854408]))

    
    normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.03693341, -0.00503048,  0.05477703]),
                                                                         variance=np.array([2.7157836, 2.7571144, 2.668294]))

    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.14760466, -0.20490389, -0.13061523]),
    #                                                                     variance=np.array([2.6515226, 2.686187 , 2.6000836]))

    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.12216299, -0.17707355, -0.10506448]),
    #                                                                     variance=np.array([2.6579645, 2.6919966, 2.6076467]))
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.15941712, -0.21857856, -0.14103174]),
    #                                                                     variance=np.array([2.650704 , 2.686771 , 2.5989208]))
    
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.14348643, -0.19796225, -0.12978919]),
    #                                                                     variance=np.array([2.6607969, 2.693048 , 2.6120377]))
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([2178.1428, 2030.9546, 2020.1321]), variance=np.array([2.0308804e+09, 1.1955764e+09, 1.0291866e+09]))

    #normalize.adapt(getTrainData(args))
    #print("mean", normalize.mean)
    #print("variance", normalize.variance)
    
    print("mean", normalize.mean_val)
    print("variance", normalize.variance_val)
    #model = xceptionnet.Xception(include_top=False, input_shape=INPUT_SHAPE)
    #x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    #x = tf.keras.layers.Dropout(0.2)(x)
    #outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    #new_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    #model = new_model
    model = xception_fused_bin.build_model(normalize=normalize)

    #model = xception_fused_bin_dwt.build_model()
    #model = xceptionnet_ablation.build_model(normalize=normalize)
    #model = xceptionnet_dwt.Xception(include_top=True, weights=None, input_shape=INPUT_SHAPE,classes=1,
    #                                         classifier_activation='sigmoid')
    learning_rate = 2e-4
    #loss = tf.keras.losses.sparse_categorical_crossentropy
    #metrics = ["accuracy"]
    loss = tf.keras.losses.binary_crossentropy
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.AUC(num_thresholds=500)]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
    
    load_status = model.load_weights("/workspace/data/trained_models/checkpoint/all_c23/ckpt.h5")
    #model.summary()
    print("Cross Stitch 1", model.get_layer(name="cross_stitch_1").weights)
    print("Cross Stitch 2", model.get_layer(name="cross_stitch_2").weights)
    print("Cross Stitch 3", model.get_layer(name="cross_stitch_3").weights)
    print("Cross Stitch 4", model.get_layer(name="cross_stitch_4").weights)
    
    model.evaluate(test_dataset, steps=TEST_SIZE // BATCH_SIZE)


def fit_on_pretrained(args):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    train_dataset = load_tfrecord(args.TRAIN_DATASET)
    val_dataset = load_tfrecord(args.VAL_DATASET)

    filepath = "/workspace/data/trained_models/checkpoint/grad_cam/base/F2F/ckpt-{epoch:02d}.h5"
    model_dir = args.MODEL_DIR
    with mirrored_strategy.scope():
        #base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE,
        #                                            classifier_activation='softmax')
        base_model = xceptionnet.Xception(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE) 
        

        base_model.trainable = False
        model = build_xception(base_model, input_shape=INPUT_SHAPE)
        learning_rate = 2e-4
        #loss = tf.keras.losses.binary_crossentropy
        #metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(), "accuracy"]
  
        loss = tf.keras.losses.sparse_categorical_crossentropy
        metrics = ["accuracy"]
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        model.summary()
        
        model.fit(train_dataset, epochs=3, steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
                  validation_data=val_dataset,
                  validation_steps=VAL_SIZE // BATCH_SIZE,
                  callbacks=None)

        model.trainable = True
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=False),
            #tf.keras.callbacks.ModelCheckpoint(
            #    filepath="/workspace/data/trained_models/checkpoint/ckpt_after_finetune_DF_c400_full_loss.h5",
            #    save_weights_only=True,
            #    monitor='val_loss',
            #    mode='min',
            #    save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=3,
                factor=0.2,
                min_lr=1e-6,
                verbose=1
            )
        ]

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
                      loss=loss,
                      metrics=metrics)
        
        model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
                  validation_data=val_dataset,
                  validation_steps=VAL_SIZE // BATCH_SIZE,
                  callbacks=callbacks)
        _, eval_accuracy = model.evaluate(
            val_dataset, steps=VAL_SIZE // BATCH_SIZE, verbose=0)
        

def load_weights_dct(args):
    test_dataset = load_tfrecord(args.TEST_DATASET, train=False)
    normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.03693341, -0.00503048,  0.05477703]),
                                                                         variance=np.array([2.7157836, 2.7571144, 2.668294]))
    print("mean", normalize.mean_val)
    print("variance", normalize.variance_val)
    base_model = xceptionnet_dct.Xception(include_top=False, input_shape=INPUT_SHAPE,
                                                    normalize=normalize)

    model = build_xception(base_model, input_shape=INPUT_SHAPE)
    

    learning_rate = 1e-5
    
    loss = tf.keras.losses.binary_crossentropy
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.AUC()]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

    load_status = model.load_weights("/workspace/data/trained_models/checkpoint/all_c23_dct/ckpt-15.h5")
    model.summary()
    model.evaluate(test_dataset, steps=TEST_SIZE // BATCH_SIZE)

        
def fit_pretrained_dct(args):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    train_dataset = load_tfrecord(args.TRAIN_DATASET)
    val_dataset = load_tfrecord(args.VAL_DATASET)
    #normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([0.03693341, -0.00503048,  0.05477703]),
    #                                                                     variance=np.array([2.7157836, 2.7571144, 2.668294]))
    
    normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=np.array([-0.14348643, -0.19796225, -0.12978919]),
                                                                         variance=np.array([2.6607969, 2.693048 , 2.6120377]))


    filepath = "/workspace/data/trained_models/checkpoint/c40_ablation/dct/ckpt-{epoch:02d}.h5"
    model_dir = args.MODEL_DIR
    with mirrored_strategy.scope():
        base_model = xceptionnet_dct.Xception(include_top=False, input_shape=INPUT_SHAPE,
                                                    normalize=normalize)

        model = build_xception(base_model, input_shape=INPUT_SHAPE)
        base_model.trainable = False
        learning_rate = 2e-4
        loss = tf.keras.losses.binary_crossentropy
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=False),
             tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=2,
                factor=0.2,
                min_lr=1e-6,
                verbose=1
            )
        ]

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        model.summary()
        model.fit(train_dataset, epochs=3, steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
                  validation_data=val_dataset,
                  validation_steps=VAL_SIZE // BATCH_SIZE,
                  callbacks=None)

        model.trainable = True
        
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
                      loss=loss,
                      metrics=metrics)
        model.summary()

        model.fit(train_dataset, epochs=100, steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
                  validation_data=val_dataset,
                  validation_steps=VAL_SIZE // BATCH_SIZE,
                  callbacks=callbacks)


def main(args):
    #args.grayscale = True
    if args.mode == "train":
        train_and_save_model(args)
        #fit_on_pretrained(args)
        #fit_pretrained_dct(args)
    elif args.mode == "test":
        load_weights(args)
        #load_weights_dct(args)
        #load_weights_pretrained(args)
    else:
        raise NotImplementedError("Specified non valid mode!")

def parse_args():
    global BATCH_SIZE, INPUT_SHAPE, CLASSES, CHANNEL_DIM
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", "-s", help="Images to load.", type=int, default=None)

    commands = parser.add_subparsers(help="Mode {train|test}.", dest="mode")

    train = commands.add_parser("train")
    epochs = 50
    train.add_argument(
        "MODEL", help="Select model to train {resnet, cnn, nn, log, log1, log2, log3}.", type=str)
    train.add_argument("TRAIN_DATASET", help="Dataset to load.", type=str)
    train.add_argument("VAL_DATASET", help="Dataset to load.", type=str)
    train.add_argument("MODEL_DIR", type=str)
    train.add_argument("CKPT_DIR", type=str)
    train.add_argument("--debug", "-d", help="Debug mode.",
                       action="store_true")
    train.add_argument(
        "--epochs", "-e", help=f"Epochs to train for; Default: {epochs}.", type=int, default=epochs)
    train.add_argument("--image_size",
                       help=f"Image size. Default: {INPUT_SHAPE}", type=int, default=128)
    train.add_argument("--early_stopping",
                       help=f"Early stopping criteria. Default: 5", type=int, default=5)
    train.add_argument("--classes",
                       help=f"Classes. Default: {CLASSES}", type=int, default=CLASSES)
    train.add_argument("--grayscale", "-g",
                       help=f"Train on grayscaled images.", action="store_true")
    train.add_argument("--batch_size", "-b",
                       help=f"Batch size. Default: {BATCH_SIZE}", type=int, default=BATCH_SIZE)
    train.add_argument("--l1",
                       help=f"L1 reguralizer intensity. Default: 0.01", type=float, default=0.01)
    train.add_argument("--l2",
                       help=f"L2 reguralizer intensity. Default: 0.01", type=float, default=0.01)


    test = commands.add_parser("test")
    test.add_argument("MODEL", help="Path to model.", type=str)
    test.add_argument("TEST_DATASET", help="Dataset to load.", type=str)
    test.add_argument("--image_size",
                      help=f"Image size. Default: {INPUT_SHAPE}", type=int, default=128)
    test.add_argument("--grayscale", "-g",
                      help=f"Test on grayscaled images.", action="store_true")
    test.add_argument("--batch_size", "-b",
                      help=f"Batch size. Default: {BATCH_SIZE}", type=int, default=BATCH_SIZE)

    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    #if args.grayscale:
    #    CHANNEL_DIM = 1

    #INPUT_SHAPE = [args.image_size, args.image_size, CHANNEL_DIM]

    if "classes" in args:
        CLASSES = args.classes

    return args

if __name__ == "__main__":
    main(parse_args())
