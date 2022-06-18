import tensorflow as tf
from tensorflow import keras

from inet.models.base_model import TaskModel


class BoundingBoxRegressionLayer(keras.layers.Layer):
    def __init__(self, backbone):
        super().__init__()
        self.image_input_shape = backbone.input.shape
        self.feature_input_shape = backbone.output.shape[1:3]
        self.regressor = self.__build_regression_stack(backbone)
        self.backbone = backbone

    def __build_regression_stack(self, backbone, dense_neurons=128, dropout_factor=0.5, num_classes=4, output_activation=None, regularization_factor=0.01):
        return keras.models.Sequential(backbone.layers + [
            keras.layers.GlobalMaxPooling2D(name='regression_pooling', input_shape=self.feature_input_shape),
            keras.layers.Dropout(dropout_factor, name='regression_dropout'),
            keras.layers.Dense(dense_neurons, name='regression_dense_layer'),
            keras.layers.BatchNormalization(name='regression_bn'),
            keras.layers.ReLU(name='regression_relu'),
            keras.layers.Dense(
                num_classes,
                activation=output_activation,
                kernel_regularizer=keras.regularizers.L2(regularization_factor),
                name='regression_output'
            )
        ])

    def get_config(self):
        return {
            'backbone': self.backbone,
            'image_input_shape': self.image_input_shape,
            'feature_input_shape': self.feature_input_shape,
        }

    def call(self, inputs):
        bbs = self.regressor(inputs)

        return tf.image.crop_and_resize(
            inputs,
            bbs,
            tf.range(0, inputs.shape[0]),
            [224, 224]  # reuse input shape
        )


class SingleStageModel(TaskModel):
    def __init__(self, backbone):
        self.bounding_box_regressor = BoundingBoxRegressionLayer(backbone)
        cropped_inputs = self.bounding_box_regressor(backbone.input)
        self.classification_stack = self.__build_classification_stack(backbone)
        classification_output = self.classification_stack(cropped_inputs)

        super().__init__(inputs=[backbone.input], outputs=[classification_output])

    def __build_classification_stack(self, backbone, dense_neurons=128, dropout_factor=0.5, num_classes=5, regularization_factor=0.01):
        return keras.models.Sequential(backbone.layers + [
            keras.layers.GlobalMaxPooling2D(name='classification_pooling', input_shape=backbone.output.shape[1:]),
            keras.layers.Dropout(dropout_factor, name='classification_dropout'),
            keras.layers.Dense(dense_neurons, name='classification_dense_layer'),
            keras.layers.BatchNormalization(name='classification_bn'),
            keras.layers.ReLU(name='classification_relu'),
            keras.layers.Dense(
                num_classes,
                activation='softmax',
                kernel_regularizer=keras.regularizers.L2(regularization_factor),
                name='classification_output'
            )
        ])

    def __build_regression_stack(self, features, dense_neurons=128, dropout_factor=0.5, num_classes=4, output_activation=None, regularization_factor=0.01):
        layer_stack = keras.layers.GlobalMaxPooling2D(name='regression_pooling')(features)

        layer_stack = keras.layers.Dropout(dropout_factor, name='regression_dropout')(layer_stack)
        layer_stack = keras.layers.Dense(dense_neurons, name='regression_dense_layer')(layer_stack)
        layer_stack = keras.layers.BatchNormalization(name='regression_bn')(layer_stack)
        layer_stack = keras.layers.ReLU(name='regression_relu')(layer_stack)
        output_layer = keras.layers.Dense(
            num_classes,
            activation=output_activation,
            kernel_regularizer=keras.regularizers.L2(regularization_factor),
            name='regression_output'
        )(layer_stack)
        return output_layer

    """
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        labels, bbs = y

        with tf.GradientTape() as tape:
            bb_pred = self.bounding_box_regressor(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(bbs, bb_pred, regularization_losses=self.losses)
            clf_pred = self.classification_stack(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss += self.compiled_loss(labels, clf_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    """

    def evaluate_predictions(self, predictions, labels, features, render_samples=False) -> None:
        pass
