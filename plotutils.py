from models import VAE
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.animation import ArtistAnimation
import seaborn as sn
from utils import build_tensor


def plot_reconstructed_images(model: VAE, test_sample: tf.Tensor, filename=None, figsize=(4, 4)):
    """Forwards samples through the model and displays them."""
    mean, log_variance = model.encode(test_sample)
    z = model.reparameterize(mean, log_variance)
    predictions = model.sample(z)
    plt.figure(figsize=figsize)
    for i in range(predictions.shape[0]):
        plt.subplot(*figsize, i+1)
        plt.imshow(predictions[i], vmin=0., vmax=1.)
        plt.axis('off')
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_latent_images(model, n, digit_size=28, filename=None):
    """Plots n x n digit images decoded from the latent space."""
    norm = tf.random.normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size*n
    image_height = image_width
    image = np.zeros((image_height, image_width, 3))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size, 3))
            image[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('Off')
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_predictions(predictions, filename=None, figsize=(4, 4)):
    """Plots images given in predictions"""
    plt.figure(figsize=figsize)
    for i in range(predictions.shape[0]):
        plt.subplot(*figsize, i+1)
        plt.imshow(predictions[i], vmin=0., vmax=1.)
        plt.axis('off')

    if filename is not None:
        plt.savefig(filename, dpi=300)

    plt.show()


def save_gif(images, n=0, fname=None):
    """saves a series of images as a gif"""
    fig = plt.figure()
    plt.axis('off')
    ims = []
    for img in images:
        im = plt.imshow(img, animated=True, vmin=0., vmax=1.)
        ims.append([im])
    anim = ArtistAnimation(fig, ims, interval=100, repeat=False)
    fname = fname if fname is not None else 'figs/cav_transition/test{}.gif'.format(n)
    anim.save(fname)


def add_decision_boundary(clf, color='gray', cav_origin=None, inv_cav_dir=False):
    """
    Adds the decision boundary to the current figure
    :param clf: The classifier
    :param color: The color of the vector
    :param cav_origin: the origin of the CAV, i.e., the base-vector where the CAV starts.
    :param inv_cav_dir: Invert direction of CAV => this is done out of simplicity reasons
    """
    bias = clf.weights[1].numpy()[0]
    w1, w2 = clf.weights[0].numpy()[:, 0]
    point_a = (0, -bias/w2)
    point_b = (-bias/w1, 0)
    m = - w1/w2
    cav = np.array([-1, 1/m])
    if inv_cav_dir:
        cav *= -1.
    dx, dy = cav / np.linalg.norm(cav)
    plt.axline(point_a, xy2=point_b, color=color)
    if cav_origin is None:
        cav_origin = point_b
    plt.arrow(*cav_origin, dx, dy, color=color, width=0.02, length_includes_head=True, head_width=0.1, zorder=3)


def plot_data(model, images_to_plot, clfs, cav_origin, legend, title, save_file_path, n_items=20):
    """
    Plots 2D latent space representation of the example images_to_plot given the model.
    The clfs will be used for displaying the decision boundary.
    :param model: The model to encode the images, needs to have a 2D latent space (unchecked)
    :param images_to_plot: list of (images, color, marker) that should be encoded by the model and displayed.
    Each element is a tuple and needs to consist of (images, color, marker):
    images is a list of images to be encoded,
    color is the color for displaying in the plot,
    marker is the marker for the images in the plot
    :param clfs: list of (clf, color, inverted) to display the decision boundaries (see add_decision_boundary)
    :param cav_origin: CAV origin for the CAVs of the classifiers (see add_decision_boundary)
    :param legend: legend-strings for the plot
    :param title: title of the plot
    :param save_file_path: path to save the plot to
    :param n_items: items to display from each element in images_to_plot
    """
    for data, color, marker in images_to_plot:
        encoded = [model.reparameterize(*model.encode(x)) for x in build_tensor(data[:n_items])]
        encoded = tf.transpose(tf.squeeze(tf.convert_to_tensor(encoded)))
        plt.scatter(encoded[0], encoded[1], c=color, marker=marker, alpha=.5, s=25)

    for clf, color, inverted in clfs:
        add_decision_boundary(clf, color, cav_origin=cav_origin, inv_cav_dir=inverted)

    plt.xlim(left=-2.5, right=2.5)
    plt.ylim(bottom=-2.5, top=2.5)
    if legend:
        plt.legend(legend)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gcf().gca().set_aspect('equal')
    if save_file_path:
        plt.savefig(save_file_path, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()


def heatmap(matrix, labels, filename=None):
    """Plots a heatmap of data"""
    data = np.round(matrix, 2)
    sn.set(font_scale=1.4)
    sn.heatmap(data, vmin=0., vmax=1., xticklabels=labels, yticklabels=labels, cmap='BuGn')
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
