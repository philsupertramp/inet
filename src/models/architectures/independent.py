from typing import Tuple

import numpy as np

from src.models.architectures.tf_lite import MultiTaskModel
from src.models.tflite_methods import evaluate_interpreted_model


class IndependentModel(MultiTaskModel):
    def __init__(self, regressor, classifier, input_shape: Tuple[int, ...] = (224, 224, 3), is_tflite=False):
        self.regressor = regressor
        self.classifier = classifier
        self.image_height = input_shape[0]
        self.image_width = input_shape[1]
        self.is_tflite = is_tflite

    def predict(self, X):
        """
        This method was build under the assumption that X is an array only containing the features
        :param X:
        :return:
        """
        if self.is_tflite:
            bbs = evaluate_interpreted_model(self.regressor, X)
        else:
            bbs = self.regressor.predict(X)

        if self.is_tflite:
            clf = evaluate_interpreted_model(self.classifier, X)
            bbs = np.array(bbs).reshape((len(bbs), -1))
            clf = np.array(clf).reshape((len(clf), -1))
            return np.c_[clf, bbs]

        return np.c_[self.classifier.predict(X), bbs]
