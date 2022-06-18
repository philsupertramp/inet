from typing import Optional

import keras_tuner
import sklearn.metrics
from tensorflow import keras

from scripts.constants import CLASS_MAP
from inet.data.constants import ModelType
from inet.data.visualization import plot_confusion_matrix
from inet.models.architectures.base_model import Backbone, SingleTaskModel
from inet.models.data_structures import ModelArchitecture
from inet.models.hyper_parameter_optimization import FrozenBlockConf


class Classifier(SingleTaskModel):
    """
    Class label prediction model

    Example:
        >>> from tensorflow.keras.applications.vgg16 import VGG16
        >>> backbone = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        >>> clf = Classifier(backbone, 128, 5, True, 'My-Classifier', 0.125, 0.5, 64, FrozenBlockConf.TRAIN_ALL.value)
        >>> clf.load_weights('my_weights.h5')
        >>> clf.predict(some_input)
    """
    ## Fixed type of model
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
        ## Number neurons in FC layer
        self.dense_neurons = dense_neurons
        ## Batch size used to train the model
        self.batch_size = batch_size

    def compile(self, learning_rate: float = 1e-6, loss='categorical_crossentropy', metrics=None, *args, **kwargs):
        """
        Extended `keras.Model.compile`.
        Adds default `Adam` optimizer and Accuracy metric

        :param learning_rate: the learning rate to train with
        :param loss: the loss function to optimize
        :param metrics: additional metrics to calculate during training
        :param args: will be passed as args to parent implementation
        :param kwargs:  will be passed as kwargs to parent implementation
        :return:
        """
        if metrics is None:
            metrics = []
        ## arguments used while calling `compile`
        self.compile_args = dict(loss=loss, learning_rate=learning_rate, metrics=metrics, args=args, kwargs=kwargs)
        super().compile(
            *args,
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy', *metrics],
            **kwargs
        )

    @staticmethod
    def evaluate_predictions(predictions, labels, features, render_samples=False) -> None:
        """
        Evaluates predictions done by a classification model.

        Computes:
        - Accuracy
        - F1-Score

        :param predictions: the predictions performed by the model to evaluate
        :param labels: ground truth labels for the predictions
        :param features: input features used to perform predictions
        :param render_samples: if `True` renders confusion matrix for predictions
        :return:
        """
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
    HPO wrapper for Classifier model.

    Used Hyper parameters (HPs):
    - Dropout factor `alpha`: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    - Learning rate `learning_rate`: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    - Number frozen layers `frozen_layers`: [TRAIN_ALL, TRAIN_HALF, TRAIN_NONE]

    Example:
        >>> import keras_tuner as kt
        >>> hpo_model = ClassifierHyperModel()
        >>> kt.BayesianOptimization(
        ...    hpo_model,
        ...    objective=kt.Objective('val_accuracy', 'max'),
        ...    max_trials=36,
        ...    directory=f'./model-selection/my-model/',
        ...    project_name='proj_name',
        ...    seed=42,
        ...    overwrite=False,
        ...    num_initial_points=12
        ... )
        >>> tuner.search(
        ...     train_set=train_set.unbatch(),
        ...     validation_set=validation_set.unbatch(),
        ...     monitoring_val='val_accuracy',
        ...     epochs=50,
        ... )
    """
    ## model configuration to use when creating a new model for HPO
    model_data: Optional[ModelArchitecture] = None
    ## model weights used
    weights: str = None

    def build(self, hp):
        """
        Builds new classification model for HPO

        :param hp: current state of HPs
        :return: model for next iteration in HPO
        """
        hp_alpha = hp.Choice('alpha', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        hp_lr = hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
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

        model.compile(learning_rate=hp_lr)
        return model
