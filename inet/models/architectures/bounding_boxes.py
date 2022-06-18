from typing import Optional

import keras_tuner
from tensorflow import keras

from inet.data.constants import ModelType
from inet.data.visualization import plot_prediction_samples
from inet.losses.giou_loss import GIoULoss
from inet.models.architectures.base_model import Backbone, SingleTaskModel
from inet.models.data_structures import ModelArchitecture


class BoundingBoxRegressor(SingleTaskModel):
    """
    Bounding Box Regression model

    Example:
        >>> from tensorflow.keras.applications.mobilenet import MobileNet
        >>> backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224))
        >>> regressor = BoundingBoxRegressor(backbone, 128, True, 'my-model', 0.125, 0.5, 64, 'relu')
        >>> regressor.load_weights('my-weights.h5')
        >>> regressor.predict([some_image])
    """
    ## Fixed type of model
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
        ## GIoU-Loss function
        self.giou_loss_fn = GIoULoss()
        ## RMSE function
        self.rmse_loss_fn = keras.metrics.RootMeanSquaredError()
        ## Number dense neurons in FC layer
        self.dense_neurons = dense_neurons
        ## Batch size used for training
        self.batch_size = batch_size

    def compile(self, learning_rate: float = 1e-6, loss='mse', metrics=None, *args, **kwargs):
        """
        Extended `keras.Model.compile`.
        Adds default `Adam` optimizer and metrics RMSE & GIoU-Loss

        :param learning_rate: the learning rate to train with
        :param loss: the loss function to optimize
        :param metrics: additional metrics to calculate during training
        :param args: will be passed as args to parent implementation
        :param kwargs:  will be passed as kwargs to parent implementation
        :return:
        """
        if metrics is None:
            metrics = []

        super().compile(
            *args,
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1),
            metrics=[self.rmse_loss_fn, self.giou_loss_fn, *metrics],
            **kwargs
        )

    @staticmethod
    def evaluate_predictions(predictions, labels, features, render_samples=False) -> None:
        """
        Evaluation method for BBox-Regression.
        Calculates metrics:
        - GIoU-Loss
        - GIoU
        - RMSE

        :param predictions: predictions done by the model
        :param labels: ground truth for predictions
        :param features: used input features to perform predictions
        :param render_samples: when `True` renders up to 25 BBox prediction samples
        :return:
        """
        giou_loss_fn = GIoULoss()
        rmse_loss_fn = keras.metrics.RootMeanSquaredError()

        giou_loss = giou_loss_fn(predictions, labels).numpy()
        print(
            '=' * 35,
            '\n\tGIoU Loss:\t', giou_loss,
            '\n\tGIoU:\t', 1. - giou_loss,
            '\n\tRMSE:\t', rmse_loss_fn(predictions, labels).numpy()
        )
        if render_samples:
            plot_prediction_samples(predictions, validation_features=features, validation_bbs=labels)


class BoundingBoxHyperModel(keras_tuner.HyperModel):
    """
    HPO wrapper for Bounding Box Regression model.

    Used Hyper parameters (HPs):
    - Dropout Factor `alpha`: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    - Learning rate `learning_rate`: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    Example:
        >>> import keras_tuner as kt
        >>> hpo_model = BoundingBoxHyperModel()
        >>> tuner = kt.BayesianOptimization(
        ...    hpo_model,
        ...    objective=kt.Objective('val_loss', 'min'),
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
        ...     monitoring_val='val_loss',
        ...     epochs=50,
        ... )
    """
    ## model configuration to use when creating a new model for HPO
    model_data: Optional[ModelArchitecture] = None

    def build(self, hp):
        """
        Build model for HPO

        :param hp: hp storage
        :return: next model for HPO
        """
        hp_alpha = hp.Choice('alpha', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        hp_lr = hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

        backbone_clone = keras.models.clone_model(self.model_data.backbone)
        backbone_clone.set_weights(self.model_data.backbone.get_weights())

        model = BoundingBoxRegressor(
            backbone_clone, dense_neurons=128, include_pooling=True, name=self.model_data.name,
            regularization_factor=hp_alpha, dropout_factor=0.5, batch_size=32
        )
        model.compile(learning_rate=hp_lr, loss=GIoULoss())
        return model
