import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import fftpack

def dct(array):
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array

def fft(array):
    array = np.transpose(array, axes=[2, 0, 1])
    array = fftpack.fft2(array)
    array = np.transpose(array, axes=[1, 2, 0])
    return array

def load_image(path):
    x = Image.open(path)
    return np.asarray(x)

def normalize(image, mean, std):
    image = (image - mean) / std
    return image

def scale_image(image):
    image /= 127.5
    image -= 1.
    return image
