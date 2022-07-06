from typing import Callable, List, Optional

import keras_tuner
import numpy as np
import tensorflow as tf
from tensorflow import keras

from inet.data.constants import ModelType
from inet.helpers import extract_labels_and_features
from inet.losses.giou_loss import GIoULoss
from inet.models.architectures.base_model import Backbone, TaskModel
from inet.models.data_structures import ModelArchitecture
from inet.models.solvers.common import evaluate_solver_predictions
from inet.models.tf_lite.tflite_methods import evaluate_interpreted_model


class TwoInOneModel(TaskModel):
    """
    Two-In-One model implementation, meaning the model solves both tasks simultaneously based on a single pass through
    the backbone (CNN)

    Example:
        >>> from tensorflow.keras.applications.mobilenet import MobileNet
        >>> backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224))
        >>> solver = TwoInOneModel(backbone, 5)
        >>> solver.load_weights('my-weights.h5', by_name=True)
        >>> solver.predict([some_input])
    """
    ## Fixed type of model
    model_type = ModelType.TWO_IN_ONE.value

    def __init__(self, backbone: Backbone, num_classes: int, dense_neurons: int = 128,
                 regularization_factor: float = 1e-3, dropout_factor: float = 0.5, batch_size: float = 32):
        """

        :param backbone: backbone CNN model
        :param dense_neurons: number dense neurons for FC layer
        :param regularization_factor: L2 regularization factor for output layer
        :param dropout_factor: factor of dropout before FC layer
        :param batch_size: batch size of the dataset
        """
        keras.backend.clear_session()

        ## Batch size used to train the model
        self.batch_size = batch_size
        # define model architecture
        ## Feature extractor (CNN) output layer index
        self.feature_layer_index = len(backbone.layers)
        # fully connected layer
        layer_stack = keras.layers.GlobalMaxPooling2D(name='pre_task_pooling')(backbone.output)

        clf_stack = keras.layers.Dropout(dropout_factor, name='clf_task_dropout')(layer_stack)
        clf_stack = keras.layers.Dense(dense_neurons, name='clf_task_dense_layer')(clf_stack)
        clf_stack = keras.layers.BatchNormalization(name='clf_task_bn')(clf_stack)
        clf_stack = keras.layers.ReLU(name='clf_task_relu')(clf_stack)
        clf_output_layer = keras.layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=keras.regularizers.L2(regularization_factor),
            name='clf_task_output'
        )(clf_stack)

        reg_stack = keras.layers.Dropout(dropout_factor, name='reg_task_dropout')(layer_stack)
        reg_stack = keras.layers.Dense(dense_neurons, name='reg_task_dense_layer')(reg_stack)
        reg_stack = keras.layers.BatchNormalization(name='reg_task_bn')(reg_stack)
        reg_stack = keras.layers.ReLU(name='reg_task_relu')(reg_stack)
        reg_output_layer = keras.layers.Dense(
            4,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(regularization_factor),
            name='reg_task_output'
        )(reg_stack)

        super().__init__(inputs=backbone.inputs, outputs=[clf_output_layer, reg_output_layer])

    def compile(self, learning_rate: float = 1e-3, loss_weights: Optional[List[float]] = None,
                losses: Optional[List[float]] = None, metrics: Optional[List] = None, *args, **kwargs):
        """
        Extended `keras.Model.compile`.
        Adds default `Adam` optimizer and Accuracy metric

        :param learning_rate: the learning rate to train with
        :param loss_weights: Task individual weight, when calculating overall loss
        :param losses: the loss functions to optimize
        :param metrics: additional metrics to calculate during training
        :param args: will be passed as args to parent implementation
        :param kwargs:  will be passed as kwargs to parent implementation
        :return:
        """
        assert losses is None, 'Losses set by implementation!'

        if metrics is None:
            pass
            # metrics = {
            #     'clf_task_output': ['accuracy'],
            #     'reg_task_output': [GIoULoss(), RootMeanSquaredError()]
            # }
        if loss_weights is None:
            loss_weights = {
                'clf_task_output': 0.5,
                'reg_task_output': 0.5,
            }
        else:
            loss_weights = {
                'clf_task_output': loss_weights[0],
                'reg_task_output': loss_weights[1],
            }

        optimizer = kwargs.pop('optimizer', keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1))

        super().compile(
            *args,
            optimizer=optimizer,
            loss={
                'clf_task_output': 'categorical_crossentropy',
                'reg_task_output': GIoULoss(),
            },
            loss_weights=loss_weights,
            metrics=['accuracy'],
            **kwargs
        )

    @staticmethod
    def evaluate_predictions(predictions, labels, features, render_samples=False) -> None:
        """shadowed method, nothing implemented here."""
        pass

    def evaluate_model(self, validation_set, preprocessing_method: Callable = None, render_samples: bool = False) -> None:
        """
        Method to evaluate predictive power of solver.

        Computes
        - Regression:
        * GIoU-Loss
        * RMSE
        - Classification:
        * Accuracy
        * F1-Score

        :param validation_set: validation data set to use
        :param preprocessing_method: preprocessing method to apply before predicting
        :param render_samples: if `True` renders confusion matrix of classification and samples for bbox regression
        :return:
        """
        def preprocess_input(x):
            if preprocessing_method is None:
                return x
            return preprocessing_method(x)

        validation_values, validation_labels = extract_labels_and_features(validation_set)
        processed_validation_values = preprocess_input(validation_values.copy())
        predictions = self.predict(processed_validation_values)

        predicted_labels = predictions[0]
        predicted_bbs = predictions[1]
        evaluate_solver_predictions(
            predicted_labels, predicted_bbs, validation_values, validation_labels, render_samples,
            'two-in-one-model'
        )

    @classmethod
    def from_config(cls, cfg) -> 'TwoInOneModel':
        """
        Helper to bootstrap solver out of configuration dictionary

        :param cfg: configuration dictionary
        :return: new TwoInOneModel instance
        """
        cfg = cfg.get('head', {})
        backbone_clone = tf.keras.models.clone_model(cfg.get('backbone'))

        model = cls(backbone_clone, **cfg.get('head'))
        model.load_weights(cfg.get('weights'), by_name=True)

        model.compile(cfg.get('learning_rate'), losses=cfg.get('losses'))
        return model


class TwoInOneTFLite:
    """
    TFLite implementation of `TwoInOneModel`

    Example:
        >>> solver = TwoInOneModel('my-weights.tflite')
        >>> solver.predict([some_input])
    """

    def __init__(self, weights):
        """

        :param weights: identifier of tflite weights
        """
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        ## TFLite model instance
        self.model = interpreter

    def predict(self, X):
        """
        Performs prediction

        :param X: vector of input features
        :return: prediction done by the model
        """
        pred = evaluate_interpreted_model(self.model, X)
        preds = []
        for p in pred:
            preds.append([p[0][0], p[1][0]])
        pred = np.array(preds)
        return pred

    def evaluate_model(self, validation_set, preprocessing_method, render_samples: bool = False):
        """
        Method to evaluate predictive power of solver.

        Computes
        - Regression:
        * GIoU-Loss
        * RMSE
        - Classification:
        * Accuracy
        * F1-Score

        :param validation_set: validation data set to use
        :param preprocessing_method: preprocessing method to apply before predicting
        :param render_samples: if `True` renders confusion matrix of classification and samples for bbox regression
        :return:
        """

        def preprocess_input(x):
            if preprocessing_method is None:
                return x
            return preprocessing_method(x)

        validation_values, validation_labels = extract_labels_and_features(validation_set)
        processed_validation_values = preprocess_input(validation_values.copy())
        predictions = self.predict(processed_validation_values)

        predicted_labels = predictions[:, 0]
        predicted_bbs = predictions[:, 1]
        predicted_bbs = tf.constant([[*i] for i in predicted_bbs])

        evaluate_solver_predictions(
            predicted_labels, predicted_bbs, validation_values, validation_labels, render_samples,
            'tflite-two-in-one-model'
        )

    @classmethod
    def from_config(cls, cfg) -> 'TwoInOneTFLite':
        """
        Constructor to load model from config-dict

        :param cfg: configuration dictionary
        :return:
        """
        return cls(cfg.get('head').get('weights'))


class TwoInOneHyperModel(keras_tuner.HyperModel):
    """
    HPO wrapper for TwoInOne model.

    Used Hyper parameters (HPs):
    - Regularization Factor `alpha`: [1e-4, 5e-2]
    - Dropout Factor `dropout`: [0.1, 0.75]
    - Learning rate `learning_rate`: [1e-4, 1e-2]

    Example:
        >>> import keras_tuner as kt
        >>> hpo_model = TwoInOneModel()
        >>> tuner = kt.BayesianOptimization(
        ...    hpo_model,
        ...    objective=kt.Objective('val_accuracy', 'max'),
        ...    max_trials=36,
        ...    directory=f'./model-selection/my-model/',
        ...    project_name='proj_name',
        ...    seed=42,
        ...    overwrite=False,
        ...    num_initial_points=12
        ...)
        >>> tuner.search(
        ...     train_set=train_set.unbatch(),
        ...     validation_set=validation_set.unbatch(),
        ...     monitoring_val='val_accuracy',
        ...     epochs=50,
        ... )
    """
    ## Model architecture configuration
    model_data: Optional[ModelArchitecture] = None

    def build(self, hp):
        """
        Builds model of next HPO iteration

        :param hp: current HP state
        :return: next HPO model
        """
        hp_alpha = hp.Float('alpha', min_value=1e-4, max_value=5e-2)
        hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.75)
        hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=0.01)

        backbone_clone = keras.models.clone_model(self.model_data.backbone)
        backbone_clone.set_weights(self.model_data.backbone.get_weights())

        model = TwoInOneModel(
            backbone_clone, num_classes=5, dense_neurons=128,
            regularization_factor=hp_alpha, dropout_factor=hp_dropout, batch_size=32
        )
        model.compile(learning_rate=hp_lr)
        return model
