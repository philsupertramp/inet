from typing import List, Optional, Tuple

from src.data.datasets import (ImageBoundingBoxDataSet, ImageLabelDataSet,
                               ImageTwoInOneDataSet)


def _directory_to_dataset(dataset_cls, directory: str, batch_size: int = 32, img_height: int = 224,
                          img_width: int = 224, class_names: Optional[List[str]] = None):
    """
    Helper method to load directory an image data set into a tf.data.Dataset

    :param dataset_cls:
    :param directory: relative/absolute path to the data set
    :param batch_size: batch size to yield
    :param img_height: image height when loading images
    :param img_width: image width when loading images
    :param class_names: set containing class name strings
    :return:
    """
    train_set = dataset_cls(directory, img_width, img_height, 'train', batch_size=batch_size,
                            class_names=class_names).build_dataset()
    validation_set = dataset_cls(directory, img_width, img_height, 'validation', batch_size=batch_size,
                                 class_names=class_names).build_dataset()
    test_set = dataset_cls(directory, img_width, img_height, 'test', batch_size=batch_size,
                           class_names=class_names).build_dataset()

    return test_set, train_set, validation_set


def directory_to_classification_dataset(
        directory: str, batch_size: int = 32, img_height: int = 224, img_width: int = 224,
        class_names: Optional[List[str]] = None
) -> Tuple[ImageLabelDataSet, ImageLabelDataSet, ImageLabelDataSet]:
    return _directory_to_dataset(ImageLabelDataSet, directory, batch_size, img_height, img_width, class_names)


def directory_to_regression_dataset(
        directory: str, batch_size: int = 32, img_height: int = 224, img_width: int = 224,
        class_names: Optional[List[str]] = None
) -> Tuple[ImageBoundingBoxDataSet, ImageBoundingBoxDataSet, ImageBoundingBoxDataSet]:
    return _directory_to_dataset(ImageBoundingBoxDataSet, directory, batch_size, img_height, img_width, class_names)


def directory_to_two_in_one_dataset(
        directory: str, batch_size: int = 32, img_height: int = 224, img_width: int = 224,
        class_names: Optional[List[str]] = None
) -> Tuple[ImageTwoInOneDataSet, ImageTwoInOneDataSet, ImageTwoInOneDataSet]:
    return _directory_to_dataset(ImageTwoInOneDataSet, directory, batch_size, img_height, img_width, class_names)
