from typing import Optional

import keras_tuner
import sklearn.metrics
from tensorflow import keras

from scripts.constants import CLASS_MAP
from src.data.constants import ModelType
from src.data.visualization import plot_confusion_matrix
from src.models.base_model import Backbone, SingleTaskModel
from src.models.data_structures import ModelArchitecture
from src.models.hyper_parameter_optimization import FrozenBlockConf


class Classifier(SingleTaskModel):
    """
    Class label prediction model
    """
    model_type = ModelType.CLASSIFICATION.value

    def __init__(self, backbone: Backbone, dense_neurons: int, num_classes: int = 5, include_pooling: bool = False,
                 name: Optional[str] = None, regularization_factor: float = 1e-3, dropout_factor: float = 0.2,
                 batch_size: int = 32, frozen_backbone_blocks: FrozenBlockConf = FrozenBlockConf.TRAIN_NONE.value):
        """

        :param backbone: backbone model
        :param num_classes: number of output neurons
        :param dense_neurons: number dense neurons for FC layer
        :param include_pooling: use pooling prior to FC layer
        :param name: model name
        :param regularization_factor: factor for L2 regularization in output layer
        :param dropout_factor: factor of dropout that's applied in front of the FC layer
        :param batch_size: batch size of the dataset
        :param frozen_backbone_blocks: allows to freeze specific layers of a model, see `FrozenBlockConf`
        """
        FrozenBlockConf(frozen_backbone_blocks).process(backbone)
        super().__init__(backbone, num_classes=num_classes, output_activation='softmax', dense_neurons=dense_neurons,
                         include_pooling=include_pooling, name=name, regularization_factor=regularization_factor,
                         dropout_factor=dropout_factor)
        self.dense_neurons = dense_neurons
        self.batch_size = batch_size

    def compile(self, learning_rate: float = 1e-6, loss='categorical_crossentropy', metrics=None, *args, **kwargs):
        if metrics is None:
            metrics = []
        self.compile_args = dict(loss=loss, learning_rate=learning_rate, metrics=metrics, args=args, kwargs=kwargs)
        super().compile(
            *args,
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy', *metrics],
            **kwargs
        )

    def evaluate_predictions(self, predictions, labels, features, render_samples=False):
        if sum(predictions.shape) > len(predictions):
            pred = predictions.argmax(axis=1)
        else:
            pred = predictions
        if sum(labels.shape) > len(labels):
            lab = labels.argmax(axis=1)
        else:
            lab = labels
        print(
            '=' * 35,
            '\n\tAccuracy:\t', sklearn.metrics.accuracy_score(lab, pred, normalize=True),
            '\n\tf1 score:\t', sklearn.metrics.f1_score(lab, pred, average='macro'),
        )
        if render_samples:
            plot_confusion_matrix(pred, lab, list(CLASS_MAP.keys()), normalize=True)


class ClassifierHyperModel(keras_tuner.HyperModel):
    """
    HPO Wrapper for classification model
    """
    weights: str = None
    model_data: ModelArchitecture = None

    def build(self, hp):
        hp_alpha = hp.Choice('alpha', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        hp_frozen_blocks = hp.Choice(
            'frozen_blocks',
            FrozenBlockConf.choices()
        )

        backbone_clone = keras.models.clone_model(self.model_data.backbone)
        backbone_clone.set_weights(self.model_data.backbone.get_weights())

        model = Classifier(
            backbone_clone, dense_neurons=128, include_pooling=True, name=self.model_data.name,
            regularization_factor=hp_alpha, dropout_factor=0.5, batch_size=32,
            frozen_backbone_blocks=hp_frozen_blocks
        )
        if self.weights:
            model.load_weights(self.weights, by_name=True)

        hp_lr = hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

        model.compile(learning_rate=hp_lr)
        return model
