"""
Data set helpers.
Use the methods defined in `load_datasets.py` to
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf

from scripts.constants import CLASS_MAP
from src.data.constants import ImageType, LabelType


class ImageDataSet:
    """
    Base class to define general behavior of an image data set.
    """
    label_key: str = None
    output_signature: Tuple[tf.TensorSpec, Union[tf.TensorSpec, Tuple[tf.TensorSpec, ...]]] = None

    def __init__(self, parent_directory: str, img_width: int, img_height: int, set_name: Optional[str] = None,
                 batch_size: int = 32, class_names: Optional[List[str]] = None):
        """

        :param parent_directory: name of parent directory of data set
        :param img_width: width when loading images
        :param img_height: height when loading images
        :param set_name: name of the subset to load
        :param batch_size: number of elements per yielded batch
        :param class_names: list of class names to find in the dataset
        """
        self.set_name = set_name or 'train'
        self.dir_name = parent_directory
        self.current_index = 0

        self.class_names = class_names
        if class_names is None:
            self.class_names = [name.split('>')[0] for name in CLASS_MAP.keys()]

        with open(os.path.join(parent_directory, 'dataset-structure.json')) as json_file:
            self.content = json.load(json_file)[self.set_name]

        self.file_count = len(self.content)
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size

    def _parse_fn(self, filename: str, label: LabelType) -> List[Union[ImageType, LabelType]]:
        """
        Method to load image from file in given size
        :param filename: image file name to load
        :param label: the associated label in the data set
        :return:
        """
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img)
        return [tf.image.resize(img, (self.img_width, self.img_height)), label]

    def _process_ds(self, ds):
        """
        Converts data set of (str, LabelType) samples to (ImageType, LabelType).
        Batches the resulting data set into provided batch size.

        :param ds: data set to transform
        :return: batched converted data set
        """
        return ds.map(self._parse_fn).batch(self.batch_size)

    def _generator_method(self):
        """
        Generator method to yield elements from the data set
        :return:
        """
        for img_file, labels in self.content.items():
            filename = img_file.split('/')[-1]
            genus = img_file.split('_')[1]
            if genus not in self.class_names:
                continue

            img_filename = os.path.join(self.dir_name, self.set_name, genus, filename)
            label = self._get_label(labels)
            yield img_filename, label

    def _get_label(self, labels_dict):
        """
        Getter
        :param labels_dict:
        :return:
        """
        return self._convert_label(labels_dict.get(self.label_key))

    def _convert_label(self, param):
        """
        Method to convert given label into desired format/form. Requires individual implementation in child classes.
        :param param: label to transform
        :return: transformed label
        """
        raise NotImplementedError('Requires implementation in child Dataset.')

    def build_dataset(self):
        """
        Method to build the dataset
        :return: the data set wrapped inside a tf.data.Dataset instance
        """
        ds = tf.data.Dataset.from_generator(
            self._generator_method,
            output_signature=self.output_signature
        )

        return self._process_ds(ds)


class ImageLabelDataSet(ImageDataSet):
    """
    yields objects of form (image, label)
    """
    label_key = 'label'

    def __init__(self, parent_directory: str, img_width: int, img_height: int, set_name: Optional[str] = None,
                 batch_size: int = 32, class_names: Optional[List[str]] = None):
        super().__init__(parent_directory, img_width, img_height, set_name, batch_size, class_names)
        self.num_classes = len(os.listdir(os.path.join(self.dir_name, self.set_name)))

        # (pz) TODO: potentially make labels integers
        self.output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32)
        )
        self.output_shapes = ((), (self.num_classes,))

    def _convert_label(self, label):
        """
        simple one hot encoding for class labels
        :param label:
        :return:
        """
        labels = [0] * self.num_classes
        labels[CLASS_MAP.get(label)] = 1
        return labels


class ImageBoundingBoxDataSet(ImageDataSet):
    """
    yields objects of form (image, bounding box).

    Image in pixel color values (0, 255)
    Bounding Box in percentages (0, 100)
    """
    label_key = 'bbs'
    output_signature = (tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(4,), dtype=tf.float32))

    def _convert_label(self, label):
        """
        Transforms BBs in COCO format to [y, x, h, w]
        :param label:
        :return:
        """
        return [
            label['y'],
            label['x'],
            label['h'],
            label['w'],
        ]


class ImageTwoInOneDataSet(ImageDataSet):
    """
    yields objects of form (image, (label, bounding box))
    """
    def __init__(self, parent_directory: str, img_width: int, img_height: int, set_name: Optional[str] = None,
                 batch_size: int = 32, class_names: Optional[List[str]] = None):
        super().__init__(parent_directory, img_width, img_height, set_name, batch_size, class_names)
        self.num_classes = len(os.listdir(os.path.join(self.dir_name, self.set_name)))
        # (pz) TODO: potentially make labels integers
        self.output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            (
                tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32),
                tf.TensorSpec(shape=(4,), dtype=tf.float32)
            )
        )

    def _get_label(self, labels_dict: Dict):
        return self._convert_label(labels_dict)

    def _convert_label(self, label: Dict):
        """
        Extracts combined labels:
        - class labels: one-hot encoded
        - bounding boxes: COCO -> [y, x, h, w]
        :param label:
        :return:
        """
        bb = [label['bbs']['y'], label['bbs']['x'], label['bbs']['h'], label['bbs']['w']]
        labels = [0] * self.num_classes
        labels[CLASS_MAP.get(label['label'])] = 1

        return labels, bb
