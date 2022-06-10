from typing import Tuple

import numpy as np

from src.models.architectures.tf_lite import MultiTaskModel
from src.models.tflite_methods import evaluate_interpreted_model


class TwoStageModel(MultiTaskModel):
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
            bbs = np.array(bbs).reshape((len(bbs), -1))
        else:
            bbs = self.regressor.predict(X)

        cropped_images = np.array([i for i in map(self.crop_image, zip(X.copy(), bbs.copy()))])

        if self.is_tflite:
            clf = evaluate_interpreted_model(self.classifier, cropped_images)
            clf = np.array(clf).reshape((len(clf), -1))
            return np.c_[clf, bbs]

        classifications = self.classifier.predict(cropped_images)
        return np.c_[classifications, bbs]
