from __future__ import absolute_import

import numpy as np
from nose.tools import (
    assert_raises,
    assert_equal,
)

from tensorflow import keras as ks
from spykes.ml.tensorflow.sparse_filtering import SparseFiltering

# Keeps the number of training images small to reduce testing time.
NUM_TRAIN = 100


def test_sparse_filtering():
    train_images = np.random.rand(NUM_TRAIN, 28 * 28)

    # Creates a simple model.
    model = ks.models.Sequential([
        ks.layers.Dense(20, input_shape=(28 * 28,), name='a'),
        ks.layers.Dense(20, name='b'),
    ])

    # Checks the four ways to pass layers.
    sf_model = SparseFiltering(model=model)
    assert_equal(len(sf_model.layer_names), len(model.layers))
    with assert_raises(ValueError):
        sf_model = SparseFiltering(model=model, layers=1)
    sf_model = SparseFiltering(model=model, layers='a')
    assert_equal(sf_model.layer_names, ['a'])

    # Checks model compilation.
    sf_model.compile('sgd')
    assert_raises(RuntimeError, sf_model.compile, 'sgd')

    sf_model = SparseFiltering(model=model, layers=['a', 'b'])
    assert_equal(sf_model.layer_names, ['a', 'b'])

    # Checks that the submodels attribute is not available yet.
    with assert_raises(RuntimeError):
        print(sf_model.submodels)

    # Checks getting a submodel.
    with assert_raises(RuntimeError):
        sf_model.get_submodel('a')

    # Checks model freezing.
    sf_model.compile('sgd', freeze=True)
    assert_equal(len(sf_model.submodels), 2)

    # Checks getting an invalid submodel.
    with assert_raises(ValueError):
        sf_model.get_submodel('c')

    # Checks model fitting.
    h = sf_model.fit(x=train_images, epochs=1)
    assert_equal(len(h), 2)  # One history for each layer name.

    # Checks the iterable cleaning part.
    assert_raises(ValueError, sf_model._clean_maybe_iterable_param, ['a'], 'a')

    def _check_works(p):
        cleaned_v = sf_model._clean_maybe_iterable_param(p, '1337')
        assert_equal(len(cleaned_v), 2)

    _check_works('a')
    _check_works(1)
    _check_works(['a', 'b'])
