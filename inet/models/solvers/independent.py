import numpy as np

from inet.models.solvers.tf_lite import MultiTaskModel
from inet.models.tf_lite.tflite_methods import evaluate_interpreted_model


class IndependentModel(MultiTaskModel):
    """
    Object detection model using independent methods to solve the localization and classification tasks.
    A regressor predicts the location and a classifier the class label, based on the original input.

    [Similar to `TwoStageModel`]

    Example:
        >>> from tensorflow.keras.applications.mobilenet import MobileNet
        >>> from inet.models.architectures.classifier import Classifier
        >>> from inet.models.architectures.bounding_boxes import BoundingBoxRegressor
        >>> clf_backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224))
        >>> reg_backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224))
        >>> regressor = BoundingBoxRegressor(reg_backbone)
        >>> classifier = Classifier(clf_backbone)
        >>> solver = IndependentModel(regressor, classifier, (224, 224, 3), False)
    """
    ## Name of the model architecture
    model_name = 'independent-model'

    def predict(self, X):
        """
        Performs independent predictions on raw input `X`

        :param X: given input features
        :return: vector of prediction tuples [label, bounding box]
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
