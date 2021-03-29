import numpy as np
import tensorflow as tf

import utils


def preprocess_images(images) -> np.ndarray:
    """
    maps images from (0, 255) to (0, 1) , binarizes pixels to be on/off
    :param images: the images to process as a numpy array
    :return: the processed images
    """
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


def load_preprocessed_mnist(colors, digits=None) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    loads mnist images and randomly colors them with colors. If digits is specified, then only those digits will be returned.
    Color labels are not provided
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    digits = np.arange(10) if digits is None else digits
    train_mask = np.in1d(y_train, digits)
    test_mask = np.in1d(y_test, digits)

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    train_size, test_size = X_train.shape[0], X_test.shape[0]

    train_images_colored = np.empty((train_size, 28, 28, 3))
    train_labels = np.empty(train_size)
    test_images_colored = np.empty((test_size, 28, 28, 3))
    test_labels = np.empty(test_size)

    random_mask_train = np.random.choice(len(colors), train_size)
    random_mask_test = np.random.choice(len(colors), test_size)
    for num, color in enumerate(colors):
        mask = random_mask_train == num
        train_images_colored[mask] = utils.map_to_rgb(X_train[mask], color)
        train_labels[mask] = y_train[mask]

        mask = random_mask_test == num
        test_images_colored[mask] = utils.map_to_rgb(X_test[mask], color)
        test_labels[mask] = y_test[mask]

    return train_images_colored, train_labels, test_images_colored, test_labels


def concept_color_img_tol(r, r_eps, g, g_eps, b, b_eps, n_images, img_shape, seed=5, tfds_batch_size=None):
    """
    Generates images in a random but uniform color. Returns a tensorflow dataset if tfds_batch_size is specified.
    The color is in the range of (r +- r_eps, g +- g_ebs, b +- b_eps).
    """
    np.random.seed(seed)
    imgs = np.empty((n_images, *img_shape, 3))
    w, h = img_shape
    imgs[:, :, :, 0] = r + np.random.default_rng().uniform(-r_eps, r_eps, n_images * w * h).reshape((n_images, w, h))
    imgs[:, :, :, 1] = g + np.random.default_rng().uniform(-g_eps, g_eps, n_images * w * h).reshape((n_images, w, h))
    imgs[:, :, :, 2] = b + np.random.default_rng().uniform(-b_eps, b_eps, n_images * w * h).reshape((n_images, w, h))
    imgs = np.clip(imgs, 0., 255.) / 255.
    return imgs if tfds_batch_size is None else tf.data.Dataset.from_tensor_slices(imgs).batch(tfds_batch_size)


def color_concept_data(color, all_colors):
    """
    Given a color C, returns the sets P_C and N_C to be used for CAV computation.
    Specifically, this function returns two numpy arrays with digits 0-9,
    the first array contains images in the specified color,
    the other array contains images in all other colors (based on the global variable colors)
    """
    other_colors = all_colors.copy()
    other_colors.remove(color)
    _, _, color_images, _ = load_preprocessed_mnist([color], np.arange(10))
    _, _, counter_images, _ = load_preprocessed_mnist(other_colors, np.arange(10))
    return color_images, counter_images


def digit_concept_data(digit, colors):
    """
    Given a digit C, returns the sets P_C and N_C to be used for CAV computation with the concept C=digit.
    Specifically, this function returns two numpy arrays with digits randomly colored (based on the global variable colors) ,
    the first array contains images of the specified digit,
    the other array contains images of all other digits.
    """
    other_digits = np.arange(10)
    other_digits = np.delete(other_digits, digit)
    _, _, digit_images, _ = load_preprocessed_mnist(colors, [digit])
    _, _, counter_images, _ = load_preprocessed_mnist(colors, other_digits)
    return digit_images, counter_images

