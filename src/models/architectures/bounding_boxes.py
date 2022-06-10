from typing import Optional

import keras_tuner
from tensorflow import keras

from src.data.constants import ModelType
from src.data.visualization import plot_prediction_samples
from src.losses.giou_loss import GIoULoss
from src.models.base_model import Backbone, SingleTaskModel


class BoundingBoxRegressor(SingleTaskModel):
    """
    Bounding Box Regression model
    """
    model_type = ModelType.REGRESSION.value

    def __init__(self, backbone: Backbone, dense_neurons: int, include_pooling: bool = False,
                 name: Optional[str] = None, regularization_factor: float = 1e-3, dropout_factor: float = 0.5,
                 batch_size: int = 32, activation_fn: str = 'relu'):
        """

        :param backbone: backbone model
        :param dense_neurons: number dense neurons for FC layer
        :param include_pooling: uses pooling before FC layer
        :param name: name of model
        :param regularization_factor: L2 regularization factor for output layer
        :param dropout_factor: factor of dropout before FC layer
        :param batch_size: batch size of the dataset
        :param activation_fn: activation function of output layer
        """
        super().__init__(backbone, 4, activation_fn, dense_neurons=dense_neurons, include_pooling=include_pooling,
                         name=name, regularization_factor=regularization_factor, dropout_factor=dropout_factor)
        self.giou_loss_fn = GIoULoss()
        self.rmse_loss_fn = keras.metrics.RootMeanSquaredError()
        self.dense_neurons = dense_neurons
        self.batch_size = batch_size

    def compile(self, learning_rate: float = 1e-6, loss='mse', metrics=None, *args, **kwargs):
        if metrics is None:
            metrics = []

        super().compile(
            *args,
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1),
            metrics=[self.rmse_loss_fn, self.giou_loss_fn, *metrics],
            **kwargs
        )

    def evaluate_predictions(self, predictions, labels, features, render_samples=False):
        print(
            '=' * 35,
            '\n\tGIoU Loss:\t', self.giou_loss_fn(predictions, labels).numpy(),
            '\n\tRMSE Loss:\t', self.rmse_loss_fn(predictions, labels).numpy()
        )
        if render_samples:
            plot_prediction_samples(predictions, validation_features=features, validation_bbs=labels)


class BoundingBoxHyperModel(keras_tuner.HyperModel):
    """
    HPO wrapper for Bounding Box Regression model
    """
    model_data = None

    def build(self, hp):
        hp_alpha = hp.Choice('alpha', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

        backbone_clone = keras.models.clone_model(self.model_data.backbone)
        backbone_clone.set_weights(self.model_data.backbone.get_weights())

        model = BoundingBoxRegressor(
            backbone_clone, dense_neurons=128, include_pooling=True, name=self.model_data.name,
            regularization_factor=hp_alpha, dropout_factor=0.5, batch_size=32
        )

        hp_lr = hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        model.compile(learning_rate=hp_lr, loss=GIoULoss())
        return model
