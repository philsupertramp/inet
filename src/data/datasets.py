import json
import os
from typing import Any, List, Optional, Tuple, Union

import tensorflow as tf

from scripts.constants import CLASS_MAP


class DataSet:
    label_key = None
    output_signature = None

    def __init__(self, parent_directory, img_width, img_height, set_name: Optional[str] = None, batch_size: int = 32,
                 class_names: Optional[List[str]] = None):
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

    def _parse_fn(self, filename, label) -> List[Union[Any, Union[str, Tuple[float, float, float, float]]]]:
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img)
        return [tf.image.resize(img, (self.img_width, self.img_height)), label]

    def _process_ds(self, ds):
        return ds.map(self._parse_fn).batch(self.batch_size)

    def generator_method(self):
        for img_file, labels in self.content.items():
            filename = img_file.split('/')[-1]
            genus = img_file.split('_')[1]
            if genus not in self.class_names:
                continue

            img_filename = os.path.join(self.dir_name, self.set_name, genus, filename)
            label = self.get_label(labels)
            yield img_filename, label

    def get_label(self, labels_dict):
        return self.convert_label(labels_dict.get(self.label_key))

    def build_dataset(self):
        ds = tf.data.Dataset.from_generator(
            self.generator_method,
            output_signature=self.output_signature
        )

        return self._process_ds(ds)

    def convert_label(self, param):
        raise NotImplementedError('Requires implementation in child Dataset.')


class ImageDataSet(DataSet):
    """
    yields objects of form (image, label)
    """
    label_key = 'label'

    def __init__(self, parent_directory, img_width, img_height, set_name: Optional[str] = None, batch_size: int = 32,
                 class_names: Optional[List[str]] = None):
        super().__init__(parent_directory, img_width, img_height, set_name, batch_size, class_names)
        self.num_classes = len(os.listdir(os.path.join(self.dir_name, self.set_name)))

        # (pz) TODO: potentially make labels integers
        self.output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32)
        )
        self.output_shapes = ((), (self.num_classes,))

    def convert_label(self, label):
        labels = [0] * self.num_classes
        labels[CLASS_MAP.get(label)] = 1
        return labels


class BoundingBoxDataSet(DataSet):
    """
    yields objects of form (image, bounding box).

    Image in pixel color values (0, 255)
    Bounding Box in percentages (0, 100)
    """
    label_key = 'bbs'
    output_signature = (tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(4,), dtype=tf.float32))

    def convert_label(self, label):
        return [
            label['y'],
            label['x'],
            label['h'],
            label['w'],
        ]


class TwoInOneDataSet(DataSet):
    """
    yields objects of form (image, (label, bounding box))
    """
    label_key = 'bbs'

    def __init__(self, parent_directory, img_width, img_height, set_name: Optional[str] = None, batch_size: int = 32,
                 class_names: Optional[List[str]] = None):
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

    def get_label(self, labels_dict):
        return self.convert_label(labels_dict)

    def convert_label(self, label):
        bb = [label['bbs']['y'], label['bbs']['x'], label['bbs']['h'], label['bbs']['w']]
        labels = [0] * self.num_classes
        labels[CLASS_MAP.get(label['label'])] = 1

        return labels, bb
