from typing import Callable, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
from inet.helpers import extract_labels_and_features
from inet.models.architectures.bounding_boxes import BoundingBoxRegressor
from inet.models.architectures.classifier import Classifier
from inet.models.solvers.common import evaluate_solver_predictions


class MultiTaskModel:
    """
    MultiTask solver implementation

    Example:
        >>> from tensorflow.keras.applications.mobilenet import MobileNet
        >>> from inet.models.architectures.classifier import Classifier
        >>> from inet.models.architectures.bounding_boxes import BoundingBoxRegressor
        >>> clf_backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224))
        >>> reg_backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224))
        >>> regressor = BoundingBoxRegressor(reg_backbone)
        >>> classifier = Classifier(clf_backbone)
        >>> solver = MultiTaskModel(regressor, classifier, (224, 224, 3), False)
        >>> solver.predict([some_input])

    """
    model_name: Optional[str] = None

    def __init__(self, regressor, classifier, input_shape: Tuple[int, ...] = (224, 224, 3), is_tflite=False):
        """

        :param regressor: BBox Regressor model (4 outputs)
        :param classifier: Classifier model (N outputs)
        :param input_shape: used for rescaling
        :param is_tflite: indicates if model is tflite version
        """
        ## Expected input image height
        self.image_height = input_shape[0]
        ## Expected input image width
        self.image_width = input_shape[1]
        ## BBox Regression model
        self.regressor = regressor
        ## Classifier model
        self.classifier = classifier
        ## indicates if model is tflite version
        self.is_tflite = is_tflite

    def predict(self, X):
        """
        Interface definition for prediction method.

        :param X: Vector of input features to perform predictions on.
        :return: predicted outputs
        """
        raise NotImplementedError()

    def evaluate_model(self, validation_set, preprocessing_method: Callable, render_samples: bool = False) -> None:
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

        predicted_labels = predictions[:, :5]
        predicted_bbs = predictions[:, 5:]

        evaluate_solver_predictions(
            predicted_labels, predicted_bbs, validation_values, validation_labels, render_samples,
            self.model_name
        )

    def crop_image(self, elem: Tuple):
        """
        Method to crop an image.

        :param elem: Tuple of [image, bb]
        :return: cropped image
        """
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
    def create_regressor(cfg: Dict) -> BoundingBoxRegressor:
        """
        Helper to bootstrap bbreg model based on provided config dict.

        :param cfg: configuration dictionary
        :return: a bounding box regression model
        """
        backbone_clone = tf.keras.models.clone_model(cfg.get('backbone'))

        regressor = BoundingBoxRegressor(backbone_clone, **cfg.get('head'))

        regressor.load_weights(cfg.get('weights'), by_name=True)

        regressor.compile(cfg.get('learning_rate'), cfg.get('loss'))

        return regressor

    @staticmethod
    def create_classifier(cfg: Dict) -> Classifier:
        """
        Helper to bootstrap classifier based on provided config dict.

        :param cfg: configuration dictionary
        :return: a classifier model
        """
        backbone_clone = tf.keras.models.clone_model(cfg.get('backbone'))

        classifier = Classifier(backbone_clone, **cfg.get('head'))

        classifier.load_weights(cfg.get('weights'), by_name=True)

        classifier.compile(cfg.get('learning_rate'), cfg.get('loss'))

        return classifier

    @classmethod
    def from_config(cls, cfg: Dict, is_tflite: bool = False) -> 'MultiTaskModel':
        """
        Method to load `MultiTaskModel` from dictionary and essentially JSON files.

        :param cfg: dict holding solver configuration
        :param is_tflite: if `True` treats config as for a tflite solver
        :return: new created solver
        """
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
