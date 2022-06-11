"""
A collection of different base models to build different model architectures.
"""
import os
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model as KerasModel

from src.data.constants import LabelType, ModelType
from src.data.datasets import ImageDataSet
from src.helpers import extract_labels_and_features, get_train_logdir


class Model(KerasModel):
    """
    Helper class to store a human-readable name
    """
    def __init__(self, inputs, outputs, name: Optional[str] = None):
        """

        :param inputs: input parameters
        :param outputs: output parameters
        :param name: verbose name for the model
        """
        ## verbose name for the model
        self.model_name = name
        if name is None:
            self.model_name = str(type(self).__name__)

        super().__init__(inputs=inputs, outputs=outputs)


class Backbone(Model):
    """
    An alias for Backbone/Feature extractor models.
    """
    pass


class TaskModel(Model):
    """
    A model tailored to solve a task. Essentially keras.Model with helper methods and wrappers for training methods.

    Example:
        >>> from tensorflow.keras.applications.mobilenet import MobileNet
        >>> backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224))
        >>> my_model = TaskModel(inputs=[backbone.input], outputs=[backbone.output])
    """
    ## Type of model
    model_type: ModelType = None

    @classmethod
    def default_callbacks(cls, monitoring_val, verbose, model_name):
        """
        Hidden helper function to create default callbacks for fit method.

        Consists of:
        - keras.callbacks.TensorBoard
        - keras.callbacks.ModelCheckpoint
        - keras.callbacks.EarlyStopping
        - keras.callbacks.ReduceLROnPlateau
        - keras.callbacks.ProgbarLogger
        - keras.callbacks.TerminateOnNaN

        :param monitoring_val: value to monitor
        :param verbose: use verbose output
        :param model_name: name of the model
        :return: list of default callbacks
        """

        run_logdir = get_train_logdir(model_name)
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        model_checkpoint_path = os.path.join(
            '../../models/',
            f'{model_name}-{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}'
        )
        checkpoint_cb = keras.callbacks.ModelCheckpoint(model_checkpoint_path, save_best_only=True)
        stopping_cb = keras.callbacks.EarlyStopping(monitor=monitoring_val, patience=5, min_delta=1e-3,
                                                    restore_best_weights=True)
        reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
            monitor=monitoring_val, factor=0.3, patience=3, min_lr=1e-12, verbose=verbose, min_delta=1e-2
        )
        return [
            tensorboard_cb,
            checkpoint_cb,
            stopping_cb,
            reduce_lr_cb,
            keras.callbacks.ProgbarLogger('samples'),
            keras.callbacks.TerminateOnNaN()
        ]

    def fit(self, train_set, validation_set, monitoring_val, batch_size: int = 32, epochs: int = 20,
            verbose: bool = True, *args, **kwargs):
        """
        Extended `keras.Model.fit` to add default callbacks

        :param train_set: dataset for training
        :param validation_set: dataset for validation
        :param monitoring_val: value to monitor while using EarlyStopping/ModelCheckpoint callbacks
        :param batch_size: the batch size to use when training the model
        :param epochs: number of epochs used in the training process
        :param verbose: verbosity setting
        :param args: further args to pass to `keras.Model.fit`
        :param kwargs: further kwargs to pass to `keras.Model.fit`
        :return:
        """
        callbacks = kwargs.pop('callbacks', self.default_callbacks(monitoring_val, verbose, self.model_name))

        return super().fit(
            train_set.batch(batch_size),
            *args,
            validation_data=validation_set.batch(batch_size),
            batch_size=batch_size,
            validation_batch_size=batch_size,
            epochs=epochs,
            verbose=2 if verbose else 0,
            callbacks=callbacks,
            **kwargs
        )

    def extract_backbone_features(self, train_set, validation_set) -> Tuple[LabelType, LabelType]:
        """
        Processes inputs using weights from backbone feature extractor.

        :param train_set:
        :param validation_set:
        :return:
        """
        feature_extractor = keras.models.Sequential(self.layers[:self.feature_layer_index + 1])
        return feature_extractor.predict(train_set), feature_extractor.predict(validation_set)

    def evaluate_model(self, validation_set: ImageDataSet, preprocessing_method: Callable = None,
                       render_samples: bool = False) -> None:
        """
        Method to evaluate a models predictive power.

        :param validation_set: the validation data set to use
        :param preprocessing_method: optional preprocessing_method for features. Gets applied before feeding the
         features into the model
        :param render_samples: boolean flag to append visualization of samples
        :return:
        """
        def preprocess_input(x):
            if preprocessing_method is None:
                return x
            return preprocessing_method(x)

        validation_values, validation_labels = extract_labels_and_features(validation_set)
        processed_validation_values = preprocess_input(np.array(validation_values.copy()))
        preds = self.predict(processed_validation_values)

        self.evaluate_predictions(preds, validation_labels, validation_values, render_samples)

    def to_tflite(self, quantization_method: 'QuantizationMethod', train_set, test_set):
        """Converts the model to a tflite compatible model"""
        from src.models.tf_lite.convert_to_tflite import create_quantize_model
        return create_quantize_model(self, train_set, test_set, quantization_method)

    @staticmethod
    def evaluate_predictions(predictions, labels, features, render_samples=False) -> None:
        """
        Implement this method to evaluate a models predictive power individually.
        For examples see `.classifier.Classifier` or `.bounding_boxes.BoundingBoxRegressor`.

        :param predictions: predictions done by the model
        :param labels: true labels for the task
        :param features: features used to make predictions
        :param render_samples: boolean flag to append visualization of samples
        :return:
        """
        raise NotImplementedError('Requires implementation in child.')


class SingleTaskModel(TaskModel):
    """
    Dedicated model architecture to solve a single task, classification or regression.
    Appends a backbone model with
    - [optional] global max pooling layer
    - 1 dropout layer with parameter `dropout_factor`
    - 1 dense layer with `dense_neurons` number neurons
    - 1 dense block
    * batch normalization
    * ReLU activation function
    * dense layer with `num_classes` neurons and `regularization_factor` for L2 kernel regularization

    Example:
        >>> from tensorflow.keras.applications.vgg16 import VGG16
        >>> vgg_backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        >>> # building a classifier for 5 classes
        >>> clf = SingleTaskModel(vgg_backbone, 5, 'softmax')
        >>> # building a bounding box regressor (outputs: [y, x, height, width])
        >>> reg = SingleTaskModel(vgg_backbone, 4, 'relu')
    """

    def __init__(self, backbone: Backbone, num_classes: int, output_activation: str, dense_neurons: int = 128,
                 include_pooling: bool = False, name: Optional[str] = None, regularization_factor: float = 1e-3,
                 dropout_factor: float = 0.5):
        """

        :param backbone: backbone model
        :param num_classes: number of output neurons
        :param output_activation: activation function of output layer
        :param dense_neurons: number dense neurons for FC layer
        :param include_pooling: use pooling prior to FC layer
        :param name: model name
        :param regularization_factor: factor for L2 regularization in output layer
        :param dropout_factor: factor of dropout that's applied in front of the FC layer
        """
        keras.backend.clear_session()
        # define model architecture
        ## Index of feature layer (last CNN layer)
        self.feature_layer_index = len(backbone.layers) - 1
        # fully connected layer
        if include_pooling:
            layer_stack = keras.layers.GlobalMaxPooling2D(name='pre_task_pooling')(backbone.output)
            self.feature_layer_index += 1
        else:
            layer_stack = backbone.output

        layer_stack = keras.layers.Dropout(dropout_factor, name='task_dropout')(layer_stack)
        layer_stack = keras.layers.Dense(dense_neurons, name='task_dense_layer')(layer_stack)
        layer_stack = keras.layers.BatchNormalization(name='task_bn')(layer_stack)
        layer_stack = keras.layers.ReLU(name='task_relu')(layer_stack)
        output_layer = keras.layers.Dense(
            num_classes,
            activation=output_activation,
            kernel_regularizer=keras.regularizers.L2(regularization_factor),
            name='task_output'
        )(layer_stack)

        # set model config
        ## desired loss function
        self.loss_fn = None
        ## value to monitor while optimizing
        self.monitoring_val = None

        ## Stores history after training cycle
        self.history = None
        ## expected input image width
        self.image_width = backbone.input.shape[2]
        ## expected input image height
        self.image_height = backbone.input.shape[1]
        super().__init__(inputs=[backbone.inputs], outputs=[output_layer], name=name)
        ## args used when calling `compile`
        self.compile_args = {}

    @staticmethod
    def evaluate_predictions(predictions, labels, features, render_samples=False):
        """
        Helper method to evaluate predictive power of model
        :param predictions: predictions done by the model
        :param labels: true labels for the task
        :param features: features used to make predictions
        :param render_samples: boolean flag to append visualization of samples
        :return:
        """
        raise NotImplementedError('Requires implementation in child.')

    @classmethod
    def from_config(cls, cfg):
        """
        Load model from configuration
        """
        backbone_clone = keras.models.clone_model(cfg.get('backbone'))

        model = cls(backbone_clone, **cfg.get('head'))

        model.load_weights(cfg.get('weights'), by_name=True)

        model.compile(cfg.get('learning_rate'), cfg.get('loss'))

        return model
