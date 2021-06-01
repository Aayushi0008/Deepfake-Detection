import argparse
import datetime as dt
import os
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import random

np.random.seed(1337)
random.seed(1337)
tf.random.set_seed(1)

import xception
import xception_cross_stitched
from dataset import deserialize_data
from models import (build_shallow_cnn, build_xception)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# complete size
TRAIN_SIZE = 400000
VAL_SIZE = 28000
TEST_SIZE = 28000

CLASSES = 2
CHANNEL_DIM = 3
INPUT_SHAPE = [299, 299, 3]

def load_tfrecord(path, train=True, unbounded=True):
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

    with mirrored_strategy.scope():
        if args.MODEL == "resnet":
            model = tf.keras.applications.ResNet50(include_top=True, weights=None, input_shape=(224, 224, 3), classes=2,
                                         classifier_activation='softmax')
        elif args.MODEL == "cnn":
            model = build_shallow_cnn(input_shape, CLASSES)
        elif args.MODEL == "xception":
            normalize = tf.keras.layers.experimental.preprocessing.Normalization()
            normalize.adapt(getTrainData(args))
    
            print("mean", normalize.mean_val)
            print("variance", normalize.variance_val)
            model = xception_cross_stitched.build_model(normalize=normalize)
            
            #model = xception.Xception(include_top=False, input_shape=INPUT_SHAPE)
            #x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
            #x = tf.keras.layers.Dropout(0.2)(x)
            #x = tf.keras.layers.Dense(512, activation='relu', name="fc_layer")(x)
            #outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
            #new_model = tf.keras.Model(inputs=model.input, outputs=outputs)
            #model = new_model

        elif args.MODEL == "densenet":
            model = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3), classes=2)
            inputs = keras.Input(shape=(224, 224, 3))
            x = model(inputs)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dropout(0.5)(x)
            outputs = keras.layers.Dense(2, activation='softmax')(x)
            model = keras.Model(inputs, outputs)

        
        loss = tf.keras.losses.binary_crossentropy
        #loss = tf.keras.losses.sparse_categorical_crossentropy
        metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy(threshold=0.5), "accuracy"]
        lr_multipliers = {5:["cross_stitch_1", "cross_stitch_2", "cross_stitch_3", "cross_stitch_4"]}
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, dict_lr_schedule=lr_multipliers)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
    
    return model


def getTrainData(args):
    path = ""
    raw_image_dataset = tf.data.TFRecordDataset(path)
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
    model = build_model(args)
    model_dir = args.MODEL_DIR 
    ckpt_path = args.CKPT_PATH
    
    callbacks = [
            tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            ),
            #tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=2,
                factor=0.2,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=False)
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
    ckpt_path = args.CKPT_PATH
    base_model = xception.Xception(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE,
                                      classifier_activation='softmax')

    inputs = tf.keras.Input(INPUT_SHAPE)
    model = build_xception(base_model, input_shape=INPUT_SHAPE)

    learning_rate = 2e-4
    loss = tf.keras.losses.binary_crossentropy
    metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    load_status = model.load_weights(ckpt_path)
    #model.summary()
    model.evaluate(test_dataset, steps=TEST_SIZE // BATCH_SIZE)


def load_weights(args):
    test_dataset = load_tfrecord(args.TEST_DATASET, train=False)
    ckpt_path = args.CKPT_PATH
    normalize = tf.keras.layers.experimental.preprocessing.Normalization()
    normalize.adapt(getTrainData(args))
    print("mean", normalize.mean)
    print("variance", normalize.variance)
    
    #print("mean", normalize.mean_val)
    #print("variance", normalize.variance_val)
    #model = xceptionnet.Xception(include_top=False, input_shape=INPUT_SHAPE)
    #x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    #x = tf.keras.layers.Dropout(0.2)(x)
    #outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    #new_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    #model = new_model
    
    model = xception_cross_stitched.build_model(normalize=normalize)

    
    learning_rate = 2e-4
    #loss = tf.keras.losses.sparse_categorical_crossentropy
    loss = tf.keras.losses.binary_crossentropy
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.AUC(num_thresholds=500)]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    
    load_status = model.load_weights(ckpt_path)
    model.summary()

    print("Cross Stitch 1", model.get_layer(name="cross_stitch_1").weights)
    
    model.evaluate(test_dataset, steps=TEST_SIZE // BATCH_SIZE)


def fit_on_pretrained(args):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    train_dataset = load_tfrecord(args.TRAIN_DATASET)
    val_dataset = load_tfrecord(args.VAL_DATASET)

    filepath = args.CKPT_PATH
    model_dir = args.MODEL_DIR
    with mirrored_strategy.scope():
        base_model = xception.Xception(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE) 

        base_model.trainable = False
        model = build_xception(base_model, input_shape=INPUT_SHAPE)
        learning_rate = 2e-4
        loss = tf.keras.losses.binary_crossentropy
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(), "accuracy"]
  
        #loss = tf.keras.losses.sparse_categorical_crossentropy
        #metrics = ["accuracy"]
        
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
                patience=8,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=False),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=3,
                factor=0.2,
                min_lr=1e-6,
                verbose=1
            )
        ]

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Low learning rate
                      loss=loss,
                      metrics=metrics)
        
        model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
                  validation_data=val_dataset,
                  validation_steps=VAL_SIZE // BATCH_SIZE,
                  callbacks=callbacks)
        _, eval_accuracy = model.evaluate(
            val_dataset, steps=VAL_SIZE // BATCH_SIZE, verbose=0)
    
                    
def main(args):
    if args.mode == "train":
        train_and_save_model(args)
        #fit_on_pretrained(args)
    elif args.mode == "test":
        load_weights(args)
        #load_weights_pretrained(args)

def parse_args():
    global BATCH_SIZE, INPUT_SHAPE, CLASSES, CHANNEL_DIM
    parser = argparse.ArgumentParser()

    commands = parser.add_subparsers(help="Mode {train|test}.", dest="mode")

    train = commands.add_parser("train")
    epochs = 50
    train.add_argument("MODEL", type=str)
    train.add_argument("TRAIN_DATASET", type=str)
    train.add_argument("VAL_DATASET", type=str)
    train.add_argument("MODEL_DIR", type=str)
    train.add_argument("CKPT_PATH", type=str)
    train.add_argument("--epochs", "-e", type=int, default=epochs)
    train.add_argument("--image_size", type=int, default=128)
    train.add_argument("--classes", type=int, default=CLASSES)
    train.add_argument("--batch_size", "-b", type=int, default=BATCH_SIZE)


    test = commands.add_parser("test")
    test.add_argument("CKPT_PATH", type=str)
    test.add_argument("TEST_DATASET", type=str)
    test.add_argument("--image_size", type=int, default=128)
    test.add_argument("--batch_size", "-b", type=int, default=BATCH_SIZE)

    args = parser.parse_args()
    BATCH_SIZE = args.batch_size

    if "classes" in args:
        CLASSES = args.classes

    return args

if __name__ == "__main__":
    main(parse_args())
