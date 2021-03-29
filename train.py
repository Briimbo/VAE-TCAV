import numpy as np
import time
import tensorflow as tf
from tqdm.autonotebook import tqdm
from models import VAE


def train(
        model: VAE,
        train_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        begin_epoch: int = 0,
        additional_epochs: int = 100,
        status_report_interval: int = 5,
        optimizer=None
):
    """
    Trains the given VAE for (additional_epochs - begin_epoch) epochs and computes the ELBO after each (status_report_interval) epoch.
    If no optimizer is given, then Adam with learning rate 1e-4 is used.
    """
    optimizer = tf.keras.optimizers.Adam(1e-4) if optimizer is None else optimizer

    train_step_fn = get_train_step_fn()

    for epoch in tqdm(range(begin_epoch, begin_epoch + additional_epochs + 1)):
        start_time = time.time()
        for train_x in train_dataset:
            train_step_fn(model, train_x, optimizer)
        end_time = time.time()

        if epoch % status_report_interval == 0 and epoch > 0:
            print_status_report(end_time - start_time, epoch, model, test_dataset)


def get_train_step_fn():
    """
    Returns a function that executes one training step and returns the loss.

    That function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    This is a wrapper such that we can train multiple different model by
    getting a training function for each.
    """
    @tf.function
    def _train_step(model: VAE, x: tf.Tensor, optimizer: tf.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return _train_step


def compute_loss(model: VAE, data: tf.Tensor):
    """
    Forwards the data through the model and computes the ELBO (Loss)
    """
    mean, logvar = model.encode(data)
    z = model.reparameterize(mean, logvar)
    data_pred = model.decode(z)

    # tensorflow vae tutorial monte carlo estimate
    cross_ent = tf.losses.mean_squared_error(data, data_pred)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def print_status_report(duration, epoch, model, test_dataset):
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, duration))
    model.save(epoch=epoch)
