"""
Common helper methods that are required in most sub-directories of the current directory
"""

import os
import time
from typing import Optional, Tuple

import numpy as np
from tensorflow import keras

## current directory path
cur_dir = os.path.dirname(__file__)
## train logdir is located in root of project
train_logdir = os.path.join(cur_dir, '../train_logs')


def get_train_logdir(name: Optional[str] = None):
    """
    Helper function to populate the current time into a given name

    Example:
        >>> get_train_logdir('foo')
        '../train_logs/foo-run_2022_06_09-13-21-50'

    :param name: name prefix for the directory
    :return: generated directory name
    """
    if name is None:
        name = ''
    run_id = time.strftime(f'{name}-run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(train_logdir, run_id)


def copy_model(input_model, layer_range: Tuple[int, int] = (0, -1)):
    """
    Method to copy `layer_range` from a model into a new model instance.

    :param input_model: the model to copy
    :param layer_range: the range of layers to copy from the input model
    :return: a new model having `layer_range` based on the `input_model`
    """
    return keras.models.Model(inputs=input_model.input, outputs=input_model.layers[layer_range[1]].output)


def extract_labels_and_features(dataset):
    """
    Helper to extract labels and features from a given data set

    :param dataset: dataset to extract data from
    :return: tuple of features and labels
    """
    values, labels = tuple(zip(*dataset.unbatch()))
    return np.array(values), np.array(labels)
