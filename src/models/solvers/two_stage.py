import numpy as np

from src.models.solvers.tf_lite import MultiTaskModel
from src.models.tf_lite.tflite_methods import evaluate_interpreted_model


class TwoStageModel(MultiTaskModel):
    """
    Object detection model using dependent/sequential methods to solve the localization and classification tasks.
    A regressor predicts the location, the original input image gets cropped to a patch containing the extracted
    Bounding Box. Afterwards a classifier predicts the class label, based on the cropped input.

    [Similar to `IndependentModel`]
    Example:
        >>> from tensorflow.keras.applications.mobilenet import MobileNet
        >>> from src.models.architectures.classifier import Classifier
        >>> from src.models.architectures.bounding_boxes import BoundingBoxRegressor
        >>> clf_backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224))
        >>> reg_backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224))
        >>> regressor = BoundingBoxRegressor(reg_backbone)
        >>> classifier = Classifier(clf_backbone)
        >>> solver = TwoStageModel(regressor, classifier, (224, 224, 3), False)

    """
    ## Name of model architecture
    model_name = 'two-stage-model'

    def predict(self, X):
        """
        Performs dependent predictions on input `X`.

        Regressor receives raw `X` -> returns `c`
        `X` is cropped using `c` -> `X_hat`
        Classifier receives `X_hat` -> returns `y`

        :param X: vector of input images
        :return: Prediction Tuple [y, c]
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
