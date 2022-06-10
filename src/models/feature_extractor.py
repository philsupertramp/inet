from typing import Tuple

import keras_tuner
from tensorflow import keras

from src.models.architectures.base_model import Backbone
from src.models.data_structures import ModelArchitecture


class FeatureExtractor(Backbone):
    def __init__(self, input_shape: Tuple[int, int, int], num_layers: int = 1, filter_size_start: int = 16):
        inputs = keras.layers.Input(shape=input_shape)

        layer_stack = keras.layers.BatchNormalization(name='image_bn')(inputs)
        current_filter_size = filter_size_start

        for layer in range(num_layers):
            layer_stack = keras.layers.Conv2D(
                current_filter_size, (3, 3), padding='same', name=f'block_{layer + 1}_conv'
            )(layer_stack)
            layer_stack = keras.layers.BatchNormalization(name=f'block_{layer + 1}_bn')(layer_stack)
            layer_stack = keras.layers.ReLU(name=f'block_{layer + 1}_relu')(layer_stack)
            layer_stack = keras.layers.MaxPooling2D(2, strides=2, name=f'block_{layer + 1}_pooling')(layer_stack)

            current_filter_size = current_filter_size * 2

        super().__init__(inputs=[inputs], outputs=[layer_stack])


class FeatureExtractorHyperModel(keras_tuner.HyperModel):
    base_model = None

    def build(self, hp):
        hp_dense_neurons = hp.Choice('dense_neurons', values=[64, 128, 256, 512])
        hp_alpha = hp.Float('alpha', min_value=1e-4, max_value=5e-2)
        hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.75)
        hp_batch_size = hp.Choice('batch_size', [4, 8, 16, 32, 64])
        hp_num_dense_layers = hp.Int('num_layers', min_value=4, max_value=6)
        hp_filter_size_start = hp.Choice('filter_size_start', [2, 4, 8, 16, 32, 64])

        backbone = FeatureExtractor(
            (224, 224, 3), num_layers=hp_num_dense_layers, filter_size_start=hp_filter_size_start
        )

        if isinstance(self.base_model, keras_tuner.HyperModel):
            model_cls = self.base_model
            model_cls.model_data = ModelArchitecture(backbone, 'hpo-')
            model = model_cls.build(hp)
        else:
            model_cls = self.base_model

            model = model_cls(
                backbone, dense_neurons=hp_dense_neurons, include_pooling=True, name=self.base_model.__class__.__name__,
                regularization_factor=hp_alpha, dropout_factor=hp_dropout, batch_size=hp_batch_size,
            )

            hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=0.01)

            model.compile(learning_rate=hp_lr)
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.get('batch_size'),
            **kwargs
        )
