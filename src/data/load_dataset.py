import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

from scripts.constants import CLASS_MAP
from src.data.datasets import BoundingBoxDataSet, ImageDataSet, TwoInOneDataSet

BBoxLabelType = Dict[str, float]
ClassLabelType = str
LabeledFileType = Dict[str, Union[BBoxLabelType, ClassLabelType]]
LabeledFileDictType = Dict[str, Dict[str, Union[BBoxLabelType, ClassLabelType]]]


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


def datasets_from_file(filename, feature_set: Tuple[str], preprocessing_method: Callable, test_share: float = 0.2,
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


def dataset_to_file(ds, filename):
    X = ds.map(lambda x, y: x)
    y = ds.map(lambda x, y: y)
    with open(filename, 'wb') as f:
        np.save(f, X)
        np.save(f, y)


def extract_file_name(elem: Dict) -> str:
    fn = elem['data']['image']
    return fn.split('/')[-1]


def extract_label(elem: Dict) -> Tuple[BBoxLabelType, ClassLabelType]:
    """
    Extracts labels from element dictionary
    :param elem:
    :return: Bounding Box coordinates and class label: ((y_min, x_min, y_max, x_max), label)
    """
    res = elem['annotations'][0]['result'][0]['value']
    x = res.get('x')
    y = res.get('y')
    h = res.get('height')
    w = res.get('width')
    return dict(x=x, y=y, w=w, h=h), res['rectanglelabels'][0]


def load_element(elem, in_directory: str) -> Dict:
    label = extract_label(elem)
    return {
        'path': os.path.join(in_directory, extract_file_name(elem)),
        'labels': {
            'label': label[1],
            'bbs': label[0]
        }
    }


def load_labels_from_bbox_file(bbox_file: str, in_directory: str) -> LabeledFileDictType:
    with open(bbox_file) as json_file:
        content = json.load(json_file)
    files = dict()
    for elem in content:
        labeled_element = load_element(elem, in_directory)
        files[labeled_element['path']] = labeled_element['labels']

    return files


def extract_labels_and_features(dataset):
    """
    Helper to extract labels and features from a given data set

    :param dataset: dataset to extract data from
    :return: tuple of features and labels
    """
    values, labels = tuple(zip(*dataset.unbatch()))
    return np.array(values), np.array(labels)


def directory_to_dataset(dataset_cls, directory: str, batch_size: int = 32, img_height: int = 224, img_width: int = 224,
                         class_names: Optional[List[str]] = None):
    train_set = dataset_cls(directory, img_width, img_height, 'train', batch_size=batch_size,
                            class_names=class_names).build_dataset()
    validation_set = dataset_cls(directory, img_width, img_height, 'validation', batch_size=batch_size,
                                 class_names=class_names).build_dataset()
    test_set = dataset_cls(directory, img_width, img_height, 'test', batch_size=batch_size,
                           class_names=class_names).build_dataset()

    return test_set, train_set, validation_set


def directory_to_classification_dataset(directory: str, batch_size: int = 32, img_height: int = 224,
                                        img_width: int = 224, class_names: Optional[List[str]] = None):
    return directory_to_dataset(ImageDataSet, directory, batch_size, img_height, img_width, class_names)


def directory_to_regression_dataset(directory: str, batch_size: int = 32, img_height: int = 224, img_width: int = 224,
                                    class_names: Optional[List[str]] = None):
    return directory_to_dataset(BoundingBoxDataSet, directory, batch_size, img_height, img_width, class_names)


def directory_to_two_in_one_dataset(directory: str, batch_size: int = 32, img_height: int = 224, img_width: int = 224,
                                    class_names: Optional[List[str]] = None):
    return directory_to_dataset(TwoInOneDataSet, directory, batch_size, img_height, img_width, class_names)


if __name__ == '__main__':
    import psutil

    print('Used memory:', psutil.virtual_memory().percent)

    train, val, test = directory_to_classification_dataset('../../train-data')

    print('Used memory:', psutil.virtual_memory().percent)

    print(train.options())
