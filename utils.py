import numpy as np
import tensorflow as tf


def map_to_rgb(images, rgb):
    """
    maps images to an rgb color, only maps pixels that are >0.5
    :param images: numpy images in range (0,1) with 1 color channel (3d)
    :param rgb: color as a tuple, e.g. (255, 0, 0) or (1, 0, 0) for red
    :return: the rgb images as a numpy array of shape (n_images, width, height, 3) with 3 channels for rgb
    """
    *shape, _ = images.shape
    r, g, b = rgb
    if r > 1. or g > 1. or b > 1.:
        r, g, b = r / 255., g / 255., b / 255.

    images = np.repeat(images[:, :, :, np.newaxis], 3, axis=3).reshape(*shape, 3)
    images[:, :, :, 0] = np.where(images[:, :, :, 0] > 0.5, r, images[:, :, :, 0])
    images[:, :, :, 1] = np.where(images[:, :, :, 1] > 0.5, g, images[:, :, :, 1])
    images[:, :, :, 2] = np.where(images[:, :, :, 2] > 0.5, b, images[:, :, :, 2])
    return images


def build_tensor(*args, n_concept_samples=100):
    """
    builds a tensor from numpy arrays specified by args. Takes a random sample of the resulting array
    """
    arr = np.concatenate(args)
    np.random.shuffle(arr)
    return tf.data.Dataset.from_tensor_slices(arr[:n_concept_samples]).batch(1)


def dataset_to_tensor(ds: tf.data.Dataset, n_elements):
    """
    Converts a tensorflow dataset (or other iterable) to a tensor and takes the first n_elements
    """
    tensors = []
    iterator = ds.__iter__()
    while n_elements > 0:
        tensors.append(iterator.__next__()[0])
        n_elements -= 1
    return tf.convert_to_tensor(tensors)
