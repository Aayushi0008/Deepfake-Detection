import argparse
import functools
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import image_paths, serialize_data
from image_np import dct2, load_image, normalize, scale_image

def collect_paths(directory):
    directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(directory).iterdir())))

    test_images = list(sorted(image_paths(directories[0])))
    train_images = list(sorted(image_paths(directories[1])))
    val_images = list(sorted(image_paths(directories[2])))
    
    print("Size of train", len(train_images))
    print("Size of val", len(val_images))
    print("Size of test", len(test_images))
    train_dataset = train_images
    val_dataset = val_images
    test_dataset = test_images

    return train_dataset, val_dataset, test_dataset


def collect_paths_fake(directory):
    directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(directory).iterdir())))

    train_dataset = []
    val_dataset = []
    test_dataset = []

    for i, directory in enumerate(directories):
        directories = sorted(map(str, filter(
            lambda x: x.is_dir(), Path(directory).iterdir())))
        test_images = list(sorted(image_paths(directories[0])))
        train_images = list(sorted(image_paths(directories[1])))
        val_images = list(sorted(image_paths(directories[2])))

        train_dataset.extend(train_images)
        test_dataset.extend(test_images)
        val_dataset.extend(val_images)
    
    print("Size of train", len(train_dataset))
    print("Size of val", len(val_dataset))
    print("Size of test", len(test_dataset))

    return train_dataset, val_dataset, test_dataset

def collect_all_paths(dirs):
    directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(dirs).iterdir())))

    train_dataset = []
    val_dataset = []
    test_dataset = []

    for i, directory in enumerate(directories):
        #if i == 0:
        #   train, val, test = collect_paths_fake(directory)
        #else:
        train, val, test = collect_paths(directory)

        train = zip(train, [i] * len(train))
        val = zip(val, [i] * len(val))
        test = zip(test, [i] * len(test))

        train_dataset.extend(train)
        val_dataset.extend(val)
        test_dataset.extend(test)

        del train, val, test

    train_dataset = np.asarray(train_dataset)
    val_dataset = np.asarray(val_dataset)
    test_dataset = np.asarray(test_dataset)

    np.random.shuffle(train_dataset)
    np.random.shuffle(val_dataset)
    np.random.shuffle(test_dataset)

    return train_dataset, val_dataset, test_dataset


def log(array, epsilon=1e-12):
    array = np.abs(array)
    array += epsilon 
    array = np.log(array)
    return array

def convert_images(inputs, load_function, transformation_function=None, absolute_function=None, normalize_function=None):
    image, label = inputs
    image = load_function(image)
    if transformation_function is not None:
        image = transformation_function(image)

    return image, label

def create_directory_tf(output_path, images, convert_function):
    os.makedirs(output_path, exist_ok=True)

    converted_images = map(convert_function, images)
    converted_images = map(serialize_data, converted_images)

    def gen():
        i = 0
        for serialized in converted_images:
            i += 1
            print(f"\rConverted {i:06d} images!", end="")
            yield serialized

    dataset = tf.data.Dataset.from_generator(
        gen, output_types=tf.string, output_shapes=())
    filename = f"{output_path}/data.tfrecords"
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)


def tfmode(directory, encode_function, outpath):
    train_dataset, val_dataset, test_dataset = collect_all_paths(directory)
    create_directory_tf(f"{outpath}_train_tf",
                        train_dataset, encode_function)
    print(f"\nConverted train images!")
    create_directory_tf(f"{outpath}_val_tf",
                        val_dataset, encode_function)
    print(f"\nConverted val images!")
    create_directory_tf(f"{outpath}_test_tf",
                        test_dataset, encode_function)
    print(f"\nConverted test images!")


def main(args):
    output = f"{args.DIRECTORY.rstrip('/')}"
    load_function = functools.partial(
        load_image, tf=args.mode == "tfrecords")
    transformation_function = None

    if args.color:
        load_function = functools.partial(load_function, grayscale=False)
        output += "_color"

    if args.raw:
        output += "_raw"
    else:
        output += "_dct"
        transformation_function = _dct2_wrapper

        if args.log:
            # log scale only for dct coefficients
            assert args.raw is False

            transformation_function = functools.partial(
                _dct2_wrapper, log=True)
            output += "_log_scaled"
            
    encode_function = functools.partial(convert_images, load_function=load_function,
                                        transformation_function=transformation_function,
                                        absolute_function=absolute_function)
    tfmode(args.DIRECTORY, encode_function, output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DIRECTORY", help="Directory to convert.",
                        type=str)

    parser.add_argument("--raw", "-r", help="Save image data as raw image.",
                        action="store_true")
    parser.add_argument("--log", "-l", help="Log scale Images.",
                        action="store_true")
    parser.add_argument("--color", "-c", help="Compute as color instead.",
                        action="store_true")

    modes = parser.add_subparsers(
        help="Select the mode {normal|tfrecords}", dest="mode")

    _ = modes.add_parser("normal")
    _ = modes.add_parser("tfrecords")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
