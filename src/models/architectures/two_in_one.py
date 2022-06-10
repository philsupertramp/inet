from typing import Callable, List, Optional

import keras_tuner
import numpy as np
import sklearn
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.metrics import RootMeanSquaredError

from scripts.constants import CLASS_MAP
from src.data.constants import ModelType
from src.data.visualization import (plot_confusion_matrix,
                                    plot_prediction_samples)
from src.helpers import extract_labels_and_features
from src.losses.giou_loss import GIoULoss
from src.models.architectures.base_model import Backbone, TaskModel
from src.models.tflite_methods import evaluate_interpreted_model


class TwoInOneModel(TaskModel):
    model_type = ModelType.TWO_IN_ONE.value

    def __init__(self, backbone: Backbone, num_classes: int, dense_neurons: int = 128,
                 regularization_factor: float = 1e-3, dropout_factor: float = 0.5, batch_size: float = 32):
        keras.backend.clear_session()

        self.batch_size = batch_size
        # define model architecture
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
        self.lite_model = None

    def compile(self, learning_rate: float = 1e-3, loss_weights: Optional[List[float]] = None,
                losses: Optional[List[float]] = None, metrics: Optional[List] = None, *args, **kwargs):
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

    def evaluate_model(self, validation_set, preprocessing_method: Callable = None, render_samples: bool = False) -> None:
        def preprocess_input(x):
            if preprocessing_method is None:
                return x
            return preprocessing_method(x)

        validation_values, validation_labels = extract_labels_and_features(validation_set)
        processed_validation_values = preprocess_input(validation_values.copy())
        predictions = self.predict(processed_validation_values)

        predicted_labels = predictions[0]
        predicted_labels_argmax = predicted_labels.copy()
        predicted_labels_argmax = predicted_labels_argmax.argmax(axis=1)
        predicted_bbs = predictions[1]

        labels = np.array([tf.argmax(i) for i in validation_labels[:, 0]])
        bbs = tf.constant([i.numpy() for i in validation_labels[:, 1]])

        acc = sklearn.metrics.accuracy_score(labels, predicted_labels_argmax)
        f1 = sklearn.metrics.f1_score(labels, predicted_labels_argmax, average='macro')
        giou = GIoULoss()(bbs, predicted_bbs).numpy()
        rmse = RootMeanSquaredError()(bbs, predicted_bbs).numpy()

        print(
            'Classification:\n',
            '=' * 35,
            '\n\tAccuracy:\t', acc,
            '\n\tf1 score:\t', f1,
            '\nLocalization:\n',
            '=' * 35,
            '\n\tGIoU:\t', 1. - giou,
            '\n\tRMSE:\t', rmse
        )
        if render_samples:
            plot_confusion_matrix(predicted_labels_argmax, labels, list(CLASS_MAP.keys()), normalize=True)
            plt.savefig(f'two-in-one-model-confusion.eps', bbox_inches='tight', pad_inches=0)
            plt.savefig(f'two-in-one-model-confusion.png', bbox_inches='tight', pad_inches=0)
            plot_prediction_samples(predicted_bbs, predicted_labels, validation_values)
            plt.savefig(f'two-in-one-model-predictions.eps', bbox_inches='tight', pad_inches=0)
            plt.savefig(f'two-in-one-model-predictions.png', bbox_inches='tight', pad_inches=0)

    @classmethod
    def from_config(cls, cfg):
        cfg = cfg.get('head', {})
        backbone_clone = tf.keras.models.clone_model(cfg.get('backbone'))

        model = cls(backbone_clone, **cfg.get('head'))
        model.load_weights(cfg.get('weights'), by_name=True)

        model.compile(cfg.get('learning_rate'), losses=cfg.get('losses'))
        return model


class TwoInOneHyperModel(keras_tuner.HyperModel):
    model_data = None

    def build(self, hp):
        hp_alpha = hp.Float('alpha', min_value=1e-4, max_value=5e-2)
        hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.75)

        backbone_clone = keras.models.clone_model(self.model_data.backbone)
        backbone_clone.set_weights(self.model_data.backbone.get_weights())

        model = TwoInOneModel(
            backbone_clone, num_classes=5, dense_neurons=128,
            regularization_factor=hp_alpha, dropout_factor=hp_dropout, batch_size=32
        )

        hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=0.01)

        model.compile(learning_rate=hp_lr)

        return model


class TwoInOneTFLite:
    def __init__(self, weights):
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        self.model = interpreter

    def predict(self, X):
        pred = evaluate_interpreted_model(self.model, X)
        preds = []
        for p in pred:
            preds.append([p[0][0], p[1][0]])
        pred = np.array(preds)
        return pred

    def evaluate_model(self, validation_set, preprocessing_method, render_samples = False):
        def preprocess_input(x):
            if preprocessing_method is None:
                return x
            return preprocessing_method(x)

        validation_values, validation_labels = extract_labels_and_features(validation_set)
        processed_validation_values = preprocess_input(validation_values.copy())
        predictions = self.predict(processed_validation_values)

        predicted_labels = predictions[:, 0]
        predicted_labels_argmax = predicted_labels.copy()
        predicted_bbs = predictions[:, 1]

        labels = np.array([tf.argmax(i) for i in validation_labels[:, 0]])
        bbs = tf.constant([i.numpy() for i in validation_labels[:, 1]])

        predicted_labels_argmax = np.array([tf.argmax(i) for i in predicted_labels_argmax])
        predicted_bbs = tf.constant([[*i] for i in predicted_bbs])

        acc = sklearn.metrics.accuracy_score(labels, predicted_labels_argmax)
        f1 = sklearn.metrics.f1_score(labels, predicted_labels_argmax, average='macro')
        giou = GIoULoss()(bbs, predicted_bbs).numpy()
        rmse = RootMeanSquaredError()(bbs, predicted_bbs).numpy()

        print(
            'Classification:\n',
            '=' * 35,
            '\n\tAccuracy:\t', acc,
            '\n\tf1 score:\t', f1,
            '\nLocalization:\n',
            '=' * 35,
            '\n\tGIoU:\t', 1. - giou,
            '\n\tRMSE:\t', rmse
        )
        if render_samples:
            plot_confusion_matrix(predicted_labels_argmax, labels, list(CLASS_MAP.keys()), normalize=True)
            plt.savefig(f'two-in-one-model-confusion.eps', bbox_inches='tight', pad_inches=0)
            plt.savefig(f'two-in-one-model-confusion.png', bbox_inches='tight', pad_inches=0)
            plot_prediction_samples(predicted_bbs, predicted_labels, validation_values)
            plt.savefig(f'two-in-one-model-predictions.eps', bbox_inches='tight', pad_inches=0)
            plt.savefig(f'two-in-one-model-predictions.png', bbox_inches='tight', pad_inches=0)

    @classmethod
    def from_config(cls, cfg, is_tflite=True):
        return cls(cfg.get('head').get('weights'))
