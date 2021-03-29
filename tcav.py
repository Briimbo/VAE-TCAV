from collections import abc
from typing import Union, Iterable

import numpy as np
import tensorflow as tf

import utils
from metrics import PosNegAccuracy
from models import VAE


def generate_x_y(pos, neg, train_size=0.8):
    """
    Transforms positive (pos) and negative (neg) samples to be used by the binary linear classifiers
    """
    concat = tf.concat([pos, neg], 0)
    features = tf.reshape(concat, shape=(concat.shape[0], concat.shape[2]))
    labels = np.array([1.]*len(pos) + [-1.] * len(neg))
    n_train = int(concat.shape[0] * train_size)
    n_test = concat.shape[0] - n_train
    train_idx = np.array([True]*n_train + [False]*n_test)
    np.random.shuffle(train_idx)
    return features[train_idx], labels[train_idx], features[~train_idx], labels[~train_idx]


def train_binary_linear_classifier(pos, neg) -> tf.keras.Model:
    """
    Trains a binary linear classifier to separate the set of positive examples from the set of negative examples
    :param pos: the positive examples encoded in the corresponding feature space
    :param neg: the negative examples encoded in the corresponding feature space
    :return: the trained classifier
    """
    x_train, y_train, x_val, y_val = generate_x_y(pos, neg, train_size=0.8)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(pos[0].shape[1],)),
        tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=10),
            activation='tanh'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='hinge',
        metrics=[PosNegAccuracy(name='accuracy')]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping('loss', min_delta=1e-4)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=500, verbose=0, callbacks=[early_stopping])
    return model


def compute_cav(model: VAE, pos, neg):
    """
    Computes the CAV for a single concept
    """
    pos_enc = [model.reparameterize(*model.encode(x)) for x in pos]
    neg_enc = [model.reparameterize(*model.encode(x)) for x in neg]
    clf = train_binary_linear_classifier(pos_enc, neg_enc)
    cav = tf.math.l2_normalize(clf.weights[0])
    return cav, clf


def compute_tcav(model: VAE, pos_concept_a, neg_concept_a, pos_concept_b, neg_concept_b):
    """
    Computes the modified VAE-TCAV score for two concepts as the dot-product between the CAVs
    """
    pos_concept_b_enc = [model.reparameterize(*model.encode(x)) for x in pos_concept_b]
    neg_concept_b_enc = [model.reparameterize(*model.encode(x)) for x in neg_concept_b]
    concept_b_classifier = train_binary_linear_classifier(pos_concept_b_enc, neg_concept_b_enc)
    concept_b_gradient = tf.math.l2_normalize(concept_b_classifier.weights[0])

    pos_concept_a_enc = [model.reparameterize(*model.encode(x)) for x in pos_concept_a]
    neg_concept_a_enc = [model.reparameterize(*model.encode(x)) for x in neg_concept_a]
    concept_a_classifier = train_binary_linear_classifier(pos_concept_a_enc, neg_concept_a_enc)
    concept_a_gradient = tf.math.l2_normalize(concept_a_classifier.weights[0])

    tcav_score = tf.tensordot(tf.transpose(concept_b_gradient), concept_a_gradient, axes=1)
    return tcav_score, concept_a_classifier, concept_b_classifier


def compute_and_print_tcav(
        model: VAE,
        pos_concept_a: Union[np.ndarray, Iterable[np.ndarray]],
        neg_concept_a:  Union[np.ndarray, Iterable[np.ndarray]],
        pos_concept_b:  Union[np.ndarray, Iterable[np.ndarray]],
        neg_concept_b:  Union[np.ndarray, Iterable[np.ndarray]],
        n_concept_samples: int = 100,
        seed: int = 0
):
    """
    Computes and prints the TCAV score for the two concepts
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    pos_concept_a = convert_to_dataset(pos_concept_a, n_concept_samples)
    neg_concept_a = convert_to_dataset(neg_concept_a, n_concept_samples)
    pos_concept_b = convert_to_dataset(pos_concept_b, n_concept_samples)
    neg_concept_b = convert_to_dataset(neg_concept_b, n_concept_samples)

    tcav_score, clf_a, clf_b = compute_tcav(
        model,
        pos_concept_a,
        neg_concept_a,
        pos_concept_b,
        neg_concept_b)

    print("TCAV={}".format(tcav_score.numpy()[0][0]))
    print("Val Acc A = {}".format(clf_a.history.history['val_accuracy'][-1]))
    print("Val Acc B = {}".format(clf_b.history.history['val_accuracy'][-1]))

    return clf_a, clf_b, tcav_score.numpy()[0][0]


def convert_to_dataset(data:  Union[np.ndarray, Iterable[np.ndarray]], n_concept_samples: int = 100):
    """
    Converts data to tensorflow dataset
    """
    if isinstance(data, abc.Iterable) and not isinstance(data, tuple):
        return utils.build_tensor(tuple(data), n_concept_samples=n_concept_samples)
    if isinstance(data, np.ndarray):
        return utils.build_tensor(data, n_concept_samples=n_concept_samples)
    if isinstance(data, tuple):
        return utils.build_tensor(*data, n_concept_samples=n_concept_samples)
    raise Exception("Unknown type in dataset conversion")
