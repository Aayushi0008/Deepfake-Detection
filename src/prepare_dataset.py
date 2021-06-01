"""Prepare image data."""
import argparse
import functools
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import image_paths, serialize_data
from image_np import dct2, load_image, normalize, scale_image
from imagenet_utils import preprocess_input

TRAIN_SIZE = 1280
VAL_SIZE = 320
TEST_SIZE = 400

def _collect_image_paths(directory):
    images = list(sorted(image_paths(directory)))

    train_dataset = images[:TRAIN_SIZE]
    val_dataset = images[TRAIN_SIZE: TRAIN_SIZE + VAL_SIZE]
    test_dataset = images[TRAIN_SIZE +
                          VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    return train_dataset, val_dataset, test_dataset


def _welford_update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    if count is None:
        count, mean, M2 = 0, np.zeros_like(new_value), np.zeros_like(new_value)

    count += 1
    delta = new_value - mean
    mean += delta / count

    delta2 = new_value - mean
    M2 += delta * delta2

    return count, mean, M2


def _welford_finalize(existing_aggregate):
    count, mean, M2 = existing_aggregate
    mean, variance, sample_variance = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return float("nan"), float("nan"), float("nan")
    else:
        return mean, variance, sample_variance


def welford(sample):
    """Calculates the mean, variance and sample variance along the first axis of an array.
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    existing_aggregate = (None, None, None)
    for data in sample:
        existing_aggregate = _welford_update(existing_aggregate, data)
    # sample variance only for calculation
    return _welford_finalize(existing_aggregate)[:-1]


def welford_multidimensional(sample):
    """Same as normal welford but for multidimensional data, computes along the last axis.
    """
    aggregates = {}

    for data in sample:
        # for each sample update each axis seperately
        for i, d in enumerate(data):
            existing_aggregate = aggregates.get(i, (None, None, None))
            existing_aggregate = _welford_update(existing_aggregate, d)
            aggregates[i] = existing_aggregate

    means, variances = list(), list()

    # in newer python versions dicts would keep their insert order, but legacy
    for i in range(len(aggregates)):
        aggregate = aggregates[i]
        mean, variance = _welford_finalize(aggregate)[:-1]
        means.append(mean)
        variances.append(variance)

    return np.asarray(means), np.asarray(variances)


def _collect_image_paths_new(directory):
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


def _collect_image_paths_fake(directory):
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

def collect(directory):
    train_dataset = []
    val_dataset = []
    test_dataset = []
    test_images = list(sorted(image_paths(directory)))
    test_dataset.extend(test_images)
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
        #   train, val, test = _collect_image_paths_fake(directory)
        #else:
        train, val, test = _collect_image_paths_new(directory)

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


def log_scale(array, epsilon=1e-12):
    """Log scale the input array.
    """
    array = np.abs(array)
    array += epsilon  # no zero in log
    array = np.log(array)
    return array


def convert_images(inputs, load_function, transformation_function=None, absolute_function=None, normalize_function=None):
    image, label = inputs
    image = load_function(image)
    if transformation_function is not None:
        image = transformation_function(image)

    if absolute_function is not None:
        image = absolute_function(image)

    if normalize_function is not None:
        image = normalize_function(image)

    return image, label


def _dct2_wrapper(image, log=False):
    image = np.asarray(image)
    image = dct2(image)
    if log:
        image = log_scale(image)

    return image


def create_directory_np(output_path, images, convert_function):
    os.makedirs(output_path, exist_ok=True)
    converted_images = map(convert_function, images)

    labels = []
    for i, (img, label) in enumerate(converted_images):
        print(f"\rConverted {i:06d} images!", end="")
        with open(f"{output_path}/{i:06}.npy", "wb+") as f:
            np.save(f, img)

        labels.append(label)

    with open(f"{output_path}/labels.npy", "wb+") as f:
        np.save(f, labels)


def normal_mode(directory, encode_function, outpath):
    (train_dataset, val_dataset, test_dataset) = collect_all_paths(directory)
    create_directory_np(f"{outpath}_train",
                        train_dataset, encode_function)
    print(f"\nConverted train images!")
    create_directory_np(f"{outpath}_val",
                        val_dataset, encode_function)
    print(f"\nConverted val images!")
    create_directory_np(f"{outpath}_test",
                        test_dataset, encode_function)
    print(f"\nConverted test images!")


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
    # we always load images into numpy arrays
    # we additionally set a flag if we later convert to tensorflow records
    load_function = functools.partial(
        load_image, tf=args.mode == "tfrecords")
    transformation_function = None
    normalize_function = None
    absolute_function = None

    if args.color:
        load_function = functools.partial(load_function, grayscale=False)
        output += "_color"

    # dct or raw image data?
    if args.raw:
        output += "_raw"

        # normalization scales to [-1, 1]
        if args.normalize:
            normalize_function = scale_image
            output += "_normalized"

    else:
        output += "_dct"
        transformation_function = _dct2_wrapper

        if args.log:
            # log scale only for dct coefficients
            assert args.raw is False

            transformation_function = functools.partial(
                _dct2_wrapper, log=True)
            output += "_log_scaled"

        if args.abs:
            # normalize to zero mean and unit variance
            train, _, _ = collect_all_paths(args.DIRECTORY)
            train = train[:TRAIN_SIZE * len(args.DIRECTORY) * 0.1]
            images = map(lambda x: x[0], train)
            images = map(load_function, images)
            images = map(transformation_function, images)

            first = next(images)
            current_max = np.absolute(first)
            for data in images:
                max_values = np.absolute(data)
                mask = current_max > max_values
                current_max *= mask
                current_max += max_values * ~mask

            def scale_by_absolute(image):
                return image / current_max

            absolute_function = scale_by_absolute

        if args.normalize:
            # normalize to zero mean and unit variance
            train, _, _ = collect_all_paths(args.DIRECTORY)
            images = map(lambda x: x[0], train)
            images = map(load_function, images)
            images = map(transformation_function, images)
            if absolute_function is not None:
                images = map(absolute_function, images)

            mean, var = welford(images)
            std = np.sqrt(var)
            output += "_normalized"
            normalize_function = functools.partial(
                normalize, mean=mean, std=std)
    encode_function = functools.partial(convert_images, load_function=load_function,
                                        transformation_function=transformation_function,
                                        normalize_function=normalize_function,
                                        absolute_function=absolute_function)
    if args.mode == "normal":
        normal_mode(args.DIRECTORY, encode_function, output)
    elif args.mode == "tfrecords":
        tfmode(args.DIRECTORY, encode_function, output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DIRECTORY", help="Directory to convert.",
                        type=str)

    parser.add_argument("--raw", "-r", help="Save image data as raw image.",
                        action="store_true")
    parser.add_argument("--log", "-l", help="Log scale Images.",
                        action="store_true")
    parser.add_argument("--abs", "-a", help="Scale each feature by its max absolute value.",
                        action="store_true")
    parser.add_argument("--color", "-c", help="Compute as color instead.",
                        action="store_true")
    parser.add_argument("--normalize", "-n", help="Normalize data.",
                        action="store_true")

    modes = parser.add_subparsers(
        help="Select the mode {normal|tfrecords}", dest="mode")

    _ = modes.add_parser("normal")
    _ = modes.add_parser("tfrecords")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
