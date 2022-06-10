import numpy as np
import sklearn
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.metrics import RootMeanSquaredError

from scripts.constants import CLASS_MAP
from src.data.visualization import (plot_confusion_matrix,
                                    plot_prediction_samples)
from src.helpers import extract_labels_and_features
from src.losses.giou_loss import GIoULoss
from src.models.architectures.bounding_boxes import BoundingBoxRegressor
from src.models.architectures.classifier import Classifier


class MultiTaskModel:
    def evaluate_model(self, validation_set, preprocessing_method, render_samples = False):
        def preprocess_input(x):
            if preprocessing_method is None:
                return x
            return preprocessing_method(x)

        validation_values, validation_labels = extract_labels_and_features(validation_set)
        processed_validation_values = preprocess_input(validation_values.copy())
        predictions = self.predict(processed_validation_values)

        predicted_labels = predictions[:, :5]
        predicted_labels_argmax = predicted_labels.copy()
        predicted_labels_argmax = predicted_labels_argmax.argmax(axis=1)
        predicted_bbs = predictions[:, 5:]

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
            plt.savefig(f'two-stage-model-confusion.eps', bbox_inches='tight', pad_inches=0)
            plt.savefig(f'two-stage-model-confusion.png', bbox_inches='tight', pad_inches=0)
            plot_prediction_samples(predicted_bbs, predicted_labels, validation_values)
            plt.savefig(f'two-stage-model-predictions.eps', bbox_inches='tight', pad_inches=0)
            plt.savefig(f'two-stage-model-predictions.png', bbox_inches='tight', pad_inches=0)

    def crop_image(self, elem):
        image, bb = elem
        if len(bb) == 2:
            bb = bb[1]
        y, x, h, w = bb
        scaled_bb = np.array([y, x, h, w]) / 100.
        return tf.image.crop_and_resize(
            [image],
            [scaled_bb],  # bb
            [0],
            [self.image_height, self.image_width]
        )[0]

    @staticmethod
    def create_regressor(cfg):
        backbone_clone = tf.keras.models.clone_model(cfg.get('backbone'))

        regressor = BoundingBoxRegressor(backbone_clone, **cfg.get('head'))

        regressor.load_weights(cfg.get('weights'), by_name=True)

        regressor.compile(cfg.get('learning_rate'), cfg.get('loss'))

        return regressor

    @staticmethod
    def create_classifier(cfg):
        backbone_clone = tf.keras.models.clone_model(cfg.get('backbone'))

        classifier = Classifier(backbone_clone, **cfg.get('head'))

        classifier.load_weights(cfg.get('weights'), by_name=True)

        classifier.compile(cfg.get('learning_rate'), cfg.get('loss'))

        return classifier

    @classmethod
    def from_config(cls, cfg, is_tflite=False):
        if is_tflite:
            reg_interpreter = tf.lite.Interpreter(model_path=cfg.get('reg').get('weights'))
            reg_interpreter.allocate_tensors()
            regressor = reg_interpreter
            clf_interpreter = tf.lite.Interpreter(model_path=cfg.get('clf').get('weights'))
            clf_interpreter.allocate_tensors()
            classifier = clf_interpreter
            return cls(regressor, classifier, is_tflite=True)

        return cls(
            cls.create_regressor(cfg.get('reg')),
            cls.create_classifier(cfg.get('clf'))
        )
