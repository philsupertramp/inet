import os
from typing import Callable, Dict, Tuple
from uuid import UUID

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

BASE_DIR = '../train-data'


def datafile_to_dict(filename, feature_list: Tuple[str]) -> Dict:
    """
    Loads .npy data file into dictionary structure.

    :param filename: name of file to load data from
    :param feature_list: List of features to extract from the dataset
    :return: dictionary containing features
    """
    elems = dict.fromkeys(feature_list)
    with open(filename, 'rb') as f:
        for elem in feature_list:
            elems[elem] = np.load(f, allow_pickle=True)
    return elems


def datadict_to_tuple(elem_dict, feature_list: Tuple[str]):
    tensors = tuple()
    for val in feature_list:
        tensors += tuple([elem_dict[val]])
    return tensors


def labels_to_vector(labels):
    new_labels = []

    num_classes = unique_labels(labels)

    for label in labels:
        val = np.array([0] * num_classes)
        val[label] = 1
        new_labels.append(val)

    return np.array(new_labels)


def datasets_from_file(filename, feature_set: Tuple[str, ...], preprocessing_method: Callable, test_share: float = 0.2,
                       batch_size: int = 16, random_state: int = 42):
    """

    :param filename:
    :param feature_set:
    :param preprocessing_method:
    :param test_share:
    :param batch_size:
    :param random_state:
    :return:
    """
    ds = datafile_to_dict(filename, feature_set)
    labels = ds.get('labels')
    X = preprocessing_method(ds.get('X'))
    del ds

    labels = labels_to_vector(labels)

    if test_share == 0.0:
        return tf.data.Dataset.from_tensor_slices((X, labels)).shuffle(buffer_size=1024).batch(batch_size)

    split = train_test_split(np.array(X), labels, test_size=test_share, random_state=random_state)
    del X, labels

    # unpack the data split
    (X_train, X_test) = split[:2]
    (y_train, y_test) = split[2:4]

    # test set is 50:50 validation/test set
    validation_test_split = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)

    (X_test, X_valid) = validation_test_split[:2]
    (y_test, y_valid) = validation_test_split[2:4]

    train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(buffer_size=1024).batch(batch_size)
    validation_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).shuffle(buffer_size=1024).batch(batch_size)
    del X_train, y_train, X_test, y_test, X_valid, y_valid

    return train_set, test_set, validation_set


def convert_datafile_to_file_structure(filename, label_names):
    train = datasets_from_file(filename, ('X', 'labels'), lambda x: x, test_share=0.0)
    directories = [os.path.join(BASE_DIR, n) for n in label_names]
    for n in directories:
        os.makedirs(n, exist_ok=True)

    for index, (x, y) in enumerate(train):
        for batch_index in range(len(x)):
            target = directories[np.argmax(y[batch_index])]
            uuid = UUID(int=index * len(x) + batch_index)
            path = os.path.join(target, f'{uuid}.jpg')
            Image.fromarray(np.array(x[batch_index])).save(path)


if __name__ == '__main__':
    labels = {'Lepidoptera', 'Coleoptera', 'Ordonata', 'Hymenoptera', 'Hemiptera'}
    convert_datafile_to_file_structure('../data-prep-full.npy', labels)
