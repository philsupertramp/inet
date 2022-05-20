"""
Helper classes/methods to generate augmented dataset.
"""

import abc
import functools
import random as rng
from enum import Enum
from typing import (Any, Callable, Iterable, List, NewType, Optional, Tuple,
                    Union)

import tensorflow as tf

"""
Unfortunately it becomes very cryptic once tf.Tensor and np.array are involved.
Therefore single elements will receive the type Any
"""
ImageType = NewType('ImageType', Any)
ClassLabelType = NewType('ClassLabelType', Any)
BoundingBoxLabelType = NewType('BoundingBoxLabelType', Any)
Features = Iterable[Any]
Labels = Iterable[Union[ClassLabelType, BoundingBoxLabelType, Tuple[ClassLabelType, BoundingBoxLabelType]]]


class LabelType(Enum):
    NONE = 0
    SINGLE = 1
    MULTI = 2


class AugmentationMethod(abc.ABC):
    """
    Abstract Base Class for augmentation methods.

    Require implementation of `process` method, as well as passing `probability` to the constructor.
    """
    def __init__(self, probability: float = 1.0):
        """

        :param probability: probability of which the method gets used for augmentation
        """
        self.probability = probability
        self.label_value_type = None

    def __str__(self):
        return f'{self.__class__.__name__} ({self.probability})'

    def process(self, features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        """
        Processing method for `AugmentationMethod` implementations.
        Override this method to perform augmentation techniques onto `features` and `labels`
        :param features: feature vector of single sample
        :param labels: label vector of single sample (multiple labels allowed)
        :return: processed pair of (features, labels)
        """
        raise NotImplementedError('Requires implementation in child.')

    def _ignore_without_bbox_labels(func):
        """
        Function decorator for methods requiring bounding box coordinate labels.

        In case the bounding box labels are not passed, the function returns the input sample.
        :return: augmented sample or input sample
        """
        @functools.wraps(func)
        def wrap(self: 'AugmentationMethod', features, labels=None, **kwargs):
            if (features is None or labels is None) or self.label_value_type == LabelType.NONE.value:
                return features, labels
            return func(self, features, labels, **kwargs)
        return wrap

    _ignore_without_bbox_labels = staticmethod(_ignore_without_bbox_labels)


class MultiProbabilityAugmentationMethod(AugmentationMethod):
    """
    Helper class to perform probability based operation on a sample.

    Inherit from this class to implement AugmentationMethod with multiple processing methods.
    """
    shared_probabilities = []

    def __init__(self, probability: float, shared_probabilities: Optional[Tuple[float, ...]] = None):
        """

        :param probability: of which the AugmentationMethod gets used for augmentation
        :param shared_probabilities: list of probabilities for child methods in same order as `method_list`
        """
        self.shared_probabilities = shared_probabilities or self.shared_probabilities
        error_msg = f'Insufficient probability vector: {self.shared_probabilities}. Requires total probability >= 1.'
        assert sum(self.shared_probabilities) >= 1.0, error_msg

        super().__init__(probability)

    @property
    def method_list(self) -> List[Callable[[Any, Any], Tuple[Any, Any]]]:
        """
        Property of implemented child methods.

        Implement this in the child class.
        :return:
        """
        raise NotImplementedError('Requires implementation in child.')

    def process(self, features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        """
        Helper method to process a sample.

        Determines child method to use based on configured `shared_probabilities` attribute.
        Sequentially iterates over list of methods, then decides individually
        :param features: vector holding features to augment
        :param labels: label vector related to features
        :return:
        """
        for current_index, method in enumerate(self.method_list):
            r = round(rng.uniform(0, 1), 2)
            if r <= self.shared_probabilities[current_index]:
                # for debugging
                # print('\tOp-sub:', method)
                return method(features, labels)
        # for debugging
        # print('\tOp-sub:', None)
        return features, labels

    def __str__(self):
        return f'{self.__class__.__name__} [{", ".join([str(i) for i in self.shared_probabilities])}]' \
               f' ({self.probability})'


class RandomContrast(AugmentationMethod):
    """
    Method to randomly increases/decreases contrast of input image by multiplying the
    feature vector with a random value from the set `[0.5, 0.6, 0.7, ..., 1.3]`.

    No transformation of input labels.
    """
    def process(self, features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        return features * rng.randrange(50, 130, 10) / 100, labels


class RandomFlip(MultiProbabilityAugmentationMethod):
    """
    Method to randomly flip the input features horizontal or vertical.
    Flips the provided Bounding Box values accordingly.
    """
    shared_probabilities = (0.5, 0.5)

    @property
    def method_list(self) -> List[Callable[[Any, Any], Tuple[Any, Any]]]:
        """See `MultiProbabilityAugmentationMethod.method_list`"""
        return [
            self.horizontal_flip,
            self.vertical_flip
        ]

    @staticmethod
    def horizontal_flip(features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        """
        Method to flip a sample on its horizontal axis.

        :param features: features to perform flipping on
        :param labels: optional array of labels, will be transformed accordingly
        :return:
        """
        new_y = labels
        if labels is not None:
            y, x, h, w = tf.unstack(labels)
            new_y = [y, 100. - (x + w), h, w]
        return tf.image.flip_left_right(features), new_y

    @staticmethod
    def vertical_flip(features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        """
        Method to flip a sample on its vertical axis

        :param features: features to perform flipping on
        :param labels: optional array of labels, will be transformed accordingly
        :return: transformed sample
        """
        new_y = labels
        if labels is not None:
            y, x, h, w = tf.unstack(labels)
            new_y = [100.0 - (y + h), x, h, w]
        return tf.image.flip_up_down(features), new_y


class RandomChannelIntensity(MultiProbabilityAugmentationMethod):
    """
    Method to randomly change the color intensity contrast of the input image.

    No transformation of input labels.
    """
    shared_probabilities = (1/3, 1/3, 1/3)

    @property
    def method_list(self):
        """See `MultiProbabilityAugmentationMethod.method_list`"""
        return [
            RandomChannelIntensity.random_scale_n_channels,
            RandomChannelIntensity.random_scale_single_channel,
            RandomChannelIntensity.random_set_single_channel
        ]

    @staticmethod
    def random_scale_n_channels(features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        """
        Method to scale N (max 3) random channels by an individual random factor

        :param features: features to perform method on
        :param labels: optional array of labels, will **not** be transformed
        :return: transformed sample
        """
        def scale_n_channels(feat):
            channel_factors = tf.constant([rng.random(), rng.random(), rng.random()], shape=(1, 1, 3))
            return channel_factors * feat

        return scale_n_channels(features), labels

    @staticmethod
    def random_scale_single_channel(features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        """
        Method to scale one random channel by a random factor.

        :param features: features to perform method on
        :param labels: optional array of labels, will **not** be transformed
        :return: transformed sample
        """
        def scale_channel(feat):
            vals = [1, 1, 1]
            vals[rng.randint(0, 2)] = rng.random()
            rng_channel = tf.constant(vals, shape=(1, 1, 3))
            return rng_channel * feat

        return scale_channel(features), labels

    @staticmethod
    def random_set_single_channel(features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        """
        Method to set a random value for a channel.

        :param features: features to perform method on
        :param labels: optional array of labels, will **not** be transformed
        :return: transformed sample
        """
        def set_input_channel(val):
            rng_channel_index = rng.randint(0, 2)
            rng_channel = tf.random.normal((*val.shape[:2], 1), 0.5, 0.12, tf.float32)
            channels = tf.split(val, 3, -1)
            return tf.concat(channels[:rng_channel_index] + [rng_channel] + channels[rng_channel_index+1:], -1)

        return set_input_channel(features), labels


class RandomCrop(MultiProbabilityAugmentationMethod):
    """
    Method to randomly crop the input image.

    Available cropping methods:
        - left
        - right
        - top
        - bottom
        - top-left
        - bottom-right

    Under the hood the method crops `pct` percent from the area between the chosen side
    and the provided Bounding Box label.

    **Note: Requires Bounding Box labels to be passed.**
    """
    shared_probabilities = (
        0.1,  # top
        0.1,  # bottom
        0.1,  # left
        0.1,  # right
        0.5,  # top_left
        0.5,  # bottom_right
    )

    @property
    def method_list(self) -> List[Callable[[Any, Any], Tuple[Any, Any]]]:
        """See `MultiProbabilityAugmentationMethod.method_list`"""
        rng_pct = rng.randint(1, 100)
        return [
            functools.partial(self.top, pct=rng_pct),
            functools.partial(self.bottom, pct=rng_pct),
            functools.partial(self.left, pct=rng_pct),
            functools.partial(self.right, pct=rng_pct),
            functools.partial(self.top_left, pct=rng_pct),
            functools.partial(self.bottom_right, pct=rng_pct),
        ]

    @staticmethod
    def _crop(image: Features, bb: Labels) -> Features:
        """Helper method to simplify image cropping with resize"""
        return tf.image.crop_and_resize(
            [image],
            [bb],
            [0],
            [*image.shape[:2]]  # reuse input shape
        )[0]

    @AugmentationMethod._ignore_without_bbox_labels
    def top(self, features: Features, labels: Labels = None, pct=50) -> Tuple[Features, Labels]:
        """
        Crops pct % of area between image top-border and bounding box.

        :param features: features to perform cropping on
        :param labels: optional array of labels, will be transformed accordingly
        :param pct: Cut-off percentage (100 full, 0 none)
        :return: cropped sample
        """
        real_pct = pct / 100.
        y, x, h, w = tf.unstack(labels)

        new_left = y * real_pct
        new_y_min = y * (1.0 - real_pct)
        new_img_val = self._crop(features, (new_left / 100., 0, 1, 1))

        h_scale = 100. / (100. - new_left)
        return new_img_val, [new_y_min * h_scale, x, h * h_scale, w]

    @AugmentationMethod._ignore_without_bbox_labels
    def bottom(self, features: Features, labels: Labels = None, pct=50) -> Tuple[Features, Labels]:
        """
        Crops pct % of area between image top-border and bounding box.

        :param features: features to perform cropping on
        :param labels: optional array of labels, will be transformed accordingly
        :param pct: Cut-off percentage (100 full, 0 none)
        :return: cropped sample
        """
        real_pct = pct / 100.
        y, x, h, w = tf.unstack(labels)

        y_max = (y + h) / 100.
        old_y_max = y_max + (1. - y_max) * (1. - real_pct)
        new_img_val = self._crop(features, (0, 0, old_y_max, 1))

        h_scale = 1 / old_y_max
        return new_img_val, [y * h_scale, x, h*h_scale, w]

    @AugmentationMethod._ignore_without_bbox_labels
    def left(self, features: Features, labels: Labels = None, pct=50) -> Tuple[Features, Labels]:
        """
        Crops pct % of area between image top-border and bounding box.

        :param features: features to perform cropping on
        :param labels: optional array of labels, will be transformed accordingly
        :param pct: Cut-off percentage (100 full, 0 none)
        :return: cropped sample
        """
        real_pct = pct / 100.
        y, x, h, w = tf.unstack(labels)

        new_left = x * real_pct
        new_img_val = self._crop(features, (0, new_left / 100., 1, 1))

        w_scale = 1. - new_left/100.
        x_min = x * (1.-real_pct)
        return new_img_val, [y, x_min / w_scale, h, w / w_scale]

    @AugmentationMethod._ignore_without_bbox_labels
    def right(self, features: Features, labels: Labels = None, pct=50) -> Tuple[Features, Labels]:
        """
        Crops pct % of area between image top-border and bounding box.

        :param features: features to perform cropping on
        :param labels: optional array of labels, will be transformed accordingly
        :param pct: Cut-off percentage (100 full, 0 none)
        :return: cropped sample
        """
        image_width = features.shape[1]
        real_pct = pct / 100.
        y, x, h, w = tf.unstack(labels)

        old_x_max = (image_width - (100. - w) * real_pct) / image_width
        new_img_val = self._crop(features, (0, 0, 1., old_x_max))

        w_scale = 1. / old_x_max
        return new_img_val, [y, x * w_scale, h, w * w_scale]

    @AugmentationMethod._ignore_without_bbox_labels
    def top_left(self, features: Features, labels: Labels = None, pct=100) -> Tuple[Features, Labels]:
        """
        Combined cropping method, crops first left, then top part, sequentially, using same pct %.

        :param features: features to perform cropping on
        :param labels: optional array of labels, will be transformed accordingly
        :param pct: Cut-off percentage (100 full, 0 none)
        :return: cropped sample
        """
        # (pz) TODO: this is a potential bottleneck. To improve speed move to a dedicated method, performing a single
        #            crop on the input image.
        new_img, new_bb = self.left(features, labels, pct=pct)
        return self.top(new_img, new_bb, pct=pct)

    @AugmentationMethod._ignore_without_bbox_labels
    def bottom_right(self, features: Features, labels: Labels = None, pct=100) -> Tuple[Features, Labels]:
        """
        Combined cropping method, crops first right, then bottom part, sequentially, using same pct %.

        :param features: features to perform cropping on
        :param labels: optional array of labels, will be transformed accordingly
        :param pct: Cut-off percentage (100 full, 0 none)
        :return: cropped sample
        """
        # (pz) TODO: this is a potential bottleneck. To improve speed move to a dedicated method, performing a single
        #            crop on the input image.
        new_img, new_bb = self.right(features, labels, pct=pct)
        return self.bottom(new_img, new_bb, pct=pct)


class RandomRotate90(MultiProbabilityAugmentationMethod):
    """
    Method to randomly rotate left, or right.

    Transforms labels accordingly.
    """

    shared_probabilities = (0.5, 0.5)

    @property
    def method_list(self) -> List[Callable[[Any, Any], Tuple[Any, Any]]]:
        """See `MultiProbabilityAugmentationMethod.method_list`"""
        return [
            self.rotate_left,
            self.rotate_right
        ]

    @staticmethod
    def rotate_left(features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        """
        Method to rotate left.

        :param features: features to perform flipping on
        :param labels: optional array of labels, will be transformed accordingly
        :return: transformed sample
        """
        new_y = labels
        if labels is not None:
            y, x, h, w = tf.unstack(labels)
            x_max = x + w
            new_y = [100. - x_max, y, w, h]
        return tf.image.rot90(features), new_y

    @staticmethod
    def rotate_right(features: Features, labels: Labels = None) -> Tuple[Features, Labels]:
        """
        Method to rotate a sample to the right.

        :param features: features to perform flipping on
        :param labels: optional array of labels, will be transformed accordingly
        :return: transformed sample
        """
        new_y = labels
        if labels is not None:
            y, x, h, w = tf.unstack(labels)
            y_max = y + h
            new_y = [x, 100. - y_max, w, h]
        return tf.image.rot90(features, k=3), new_y


class DataAugmentationHelper:
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, number_samples: int, operations: Optional[List[AugmentationMethod]] = None,
                 batch_size: int = 32, seed: int = 42, bbox_label_index: Optional[int] = None,
                 label_value_type: LabelType = LabelType.SINGLE.value, output_signature: Optional[Tuple] = None):
        """
        Class to simplify data augmentation on a given tf.data.Dataset.


        :param number_samples: desired number of total augmented samples in the dataset
        :param operations: Array of augmentation methods to perform on the dataset
        :param batch_size: the resulting batch size of the dataset
        :param seed: a seed to reproduce randomness in the generation process
        :param bbox_label_index: index of bounding box labels,
        required if `label_value_type = LabelType.MULTI.value`
        (see `__perform_transformation` for further configuration info)
        :param label_value_type: index  (see `__perform_transformation` for further configuration info)
        :param output_signature:
        """
        self.number_samples = number_samples
        self.batch_size = batch_size
        self.bbox_label_index = bbox_label_index
        self.label_value_type = label_value_type
        self.operations = operations or []
        self.seed = seed
        self._output_signature = output_signature or (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=4, dtype=tf.float32)
        )
        self.picks = dict.fromkeys([str(op) for op in self.operations + [None]], 0)

    def __set_seed(self) -> None:
        """
        Private helper method to set environment seeds
        :return: None
        """
        rng.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __perform_transformation(self, feature_vector: Features, label_vector: Labels) -> Tuple[Features, Labels]:
        """
        Private method to perform transformation of sample.

        This method determines whether to use a provided label for augmentation or not according to provided
        `label_value_type` and `bbox_label_index` settings.

        **Note: This method determines the used operation by looping over list of registered operations,
        in case rng fails results in input sample as output.**

        Possible combinations:
            - LabelType.NONE -> (processed features, input labels)
            - LabelType.SINGLE -> (processed features, processed labels)
            - LabelType.MULTI, bbox_label_index != None -> (processed features, processed labels)
            - label_value_type = None -> (input features, input labels)

        :raises AssertionError: when using `label_value_type = LabelType.MULTI.value` without setting `bbox_label_index`

        :param feature_vector: the feature vector to perform augmentation on
        :param label_vector: related labels to the feature vector
        :return: augmented sample pair of (features, labels)
        """
        op = None

        # determine operation to use
        for operation in self.operations:
            r = round(rng.uniform(0, 1), 2)
            if r <= operation.probability:
                op = operation
        # for debugging
        # print('OP:', op)

        self.picks[str(op)] += 1

        # early exit: in case we didn't find a method return raw sample
        if op is None:
            return feature_vector, label_vector

        # No labels, ignore them
        if self.label_value_type == LabelType.NONE.value:
            new_features, _ = op.process(feature_vector, None)
            return new_features, label_vector

        # Single Label containing BBs
        elif self.label_value_type == LabelType.SINGLE.value:
            return op.process(feature_vector, label_vector)

        # Multi Label, requires bbox_label_index
        elif self.label_value_type == LabelType.MULTI.value:
            assert self.bbox_label_index is not None, 'Multi label augmentation requires setting of `bbox_label_index`!'

            new_features, new_labels = op.process(feature_vector, label_vector[self.bbox_label_index])
            return (
                new_features,
                label_vector[:self.bbox_label_index] + [new_labels] + label_vector[self.bbox_label_index + 1:]
            )

        # for debugging
        # print('OP: None')
        return feature_vector, label_vector

    def transform_generator(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Generator method yielding `number_samples` augmented samples.

        This method is essentially just a wrapper around `__perform_transformation`,
        it configures the environment and yields `number_samples` augmented samples sequentially
        using the provided dataset.
        After each cycle of the dataset it gets shuffled prior to yielding the next sample.

        :param dataset: input dataset to augment on
        :return: generator yielding augmented samples
        """
        self.__set_seed()

        yielded_element = 0
        while yielded_element < self.number_samples:
            # (pz) TODO: the line below might be not performant at all. One could technically use a
            #            buffer size equal to a fraction of or full `num_samples`. But this will
            #            temporarily keep `buffer_size` objects in memory.
            #            It's needless to stress, using a fixed value of 100 isn't any good
            #            practice and especially not dynamic enough, but I don't have the time
            #            optimizing this part, so I will keep it for now.
            #            Speaking about caching and tf.data.Dataset this blob post
            #            [^1](https://www.determined.ai/blog/tf-dataset-the-bad-parts) seems
            #            like a good entry point to learn more about this topic.
            dataset = dataset.shuffle(100)
            for sample_features, sample_labels in dataset:
                if yielded_element >= self.number_samples:
                    break

                yield self.__perform_transformation(sample_features, sample_labels)
                yielded_element += 1

    def transform(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Wrapper method to generate a tf.data.Dataset with augmentation according to passed
        `operations` array.

        :param dataset: the dataset to convert to an augmented dataset
        :return: augmented dataset
        """

        # set nested attributes
        for op in self.operations:
            op.label_value_type = self.label_value_type

        # we shall not use lambda functions for generators as they tend to become
        # too complicated to debug, hence we use a nested method instead, this allows
        # us to "bake" the dataset into the generator.
        def generator_fn():
            return self.transform_generator(dataset)

        return tf.data.Dataset.from_generator(
            generator_fn,
            output_signature=self._output_signature
        )


if __name__ == '__main__':
    """
    Example of data augmentation for a classification dataset.
    """
    import matplotlib.pyplot as plt

    from src.data.load_dataset import directory_to_classification_dataset

    ds, _, _ = directory_to_classification_dataset('../../data/iNat/cropped-data')
    augmenter = DataAugmentationHelper(
        1024,
        operations=[
            RandomContrast(probability=0.2),
            RandomFlip(probability=0.5),
            RandomRotate90(probability=0.3)
        ],
        label_value_type=LabelType.NONE.value,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=5, dtype=tf.float32)
        )
    )
    augmented_ds = augmenter.transform(ds.unbatch())

    plt.figure(figsize=(50, 50))
    i = 1
    for img, _ in augmented_ds:
        plt.subplot(32, 32, i)
        plt.imshow(img.numpy()/255.)
        plt.axis('off')
        i += 1
    plt.savefig('flip-contrast-test.png')
