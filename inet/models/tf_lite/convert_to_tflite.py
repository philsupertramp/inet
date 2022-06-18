import os
import tempfile
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from inet.models.architectures.base_model import TaskModel
from inet.models.tf_lite.tflite_methods import evaluate_q_model, validate_q_model_prediction

tf_cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization


class QuantizationMethod(Enum):
    """
    Helper enum to determine a quantization method
    """
    ## No quantization
    NONE = None
    ## Dynamic quantization
    DYNAMIC = 1
    ## Conversion to float16
    FLOAT_16 = 2
    ## Conversion to uint8
    FULL_INT = 3


class ClusterMethod(Enum):
    """
    Helper enum to determine a cluster methods
    """
    ## Linear clustering method
    LINEAR = CentroidInitialization.LINEAR
    ## Random clustering method
    RANDOM = CentroidInitialization.RANDOM
    ## Clustering based on density
    DENSITY_BASED = CentroidInitialization.DENSITY_BASED
    ## Clustering using KMeans++ algo
    KMEANS_PLUS_PLUS = CentroidInitialization.KMEANS_PLUS_PLUS


def create_q_aware_model(model):
    """
    Create quantization aware model

    :param model: model to convert
    :return: quantization aware model
    """
    quant_aware_model = tfmot.quantization.keras.quantize_model(model)

    # Save or checkpoint the model.
    _, keras_model_file = tempfile.mkstemp('.h5')
    quant_aware_model.save(keras_model_file)

    # `quantize_scope` is needed for deserializing HDF5 models.
    with tfmot.quantization.keras.quantize_scope():
        loaded_model = tf.keras.models.load_model(keras_model_file)

    loaded_model.summary()
    return loaded_model


def create_tf_lite_q_model(q_model, train_set, quant_method: QuantizationMethod = QuantizationMethod.FULL_INT, model_name='bbreg'):
    """
    converts regular model to q aware model using provided `quant_method`

    :param q_model: a quantization aware model
    :param train_set: samples representing the train set
    :param quant_method: quantization method
    :param model_name: resulting model name
    :return: tf lite version of quantization aware model
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(q_model)
    # This step is needed in all quantization strategies
    tf_file = model_name + '.tflite'
    if quant_method.value != QuantizationMethod.NONE.value:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quant_method.value == QuantizationMethod.FLOAT_16.value:  # float16-quantization
            converter.target_spec.supported_types = [tf.float16]

        elif quant_method.value == QuantizationMethod.FULL_INT.value:  # full integer quantization

            # You need to measure the dynamic range of activations and inputs
            # by supplying sample input data to the converter
            def representative_data_gen():
                for feature_batch, _ in train_set.take(5):
                    if len(feature_batch.shape) > 3:
                        for feature in feature_batch:
                            yield [feature[tf.newaxis, :]]  # Inputs to TFLite models require one extra dim.
                    else:
                        yield [feature_batch[tf.newaxis, :]]

            converter.representative_dataset = representative_data_gen

        # Define name of quantized TFLite model
        tf_file = model_name + '_' + quant_method.name + '.tflite'
    tflite_model = converter.convert()
    # Save the TFLite model
    with tf.io.gfile.GFile(tf_file, 'wb') as f:
        f.write(tflite_model)
    print('Model in Mb:', os.path.getsize(tf_file) / float(2 ** 20))
    return tflite_model


def create_quantize_model(model: TaskModel, train_set, test_set, quant_method: QuantizationMethod):
    """
    Method to create and validate a quantized version of `model` using `quant_method`.

    :param model: the model instance to quantize
    :param train_set: the train set, will be used as representation when using QuantizationMethod.FULL_INT
    :param test_set: to evaluate the models
    :param quant_method: quantization method applied onto the model
    :return: q-aware-model
    """
    test_images, test_labels = tuple(zip(*test_set.unbatch()))

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    model_prediction = model.predict(test_images, verbose=0)
    tf_lite_model = create_tf_lite_q_model(model, train_set, quant_method, model.model_name)

    tfl_model_prediction = evaluate_q_model(tf_lite_model, test_images)
    validate_q_model_prediction(model_prediction, tfl_model_prediction, test_labels, model.model_type)

    return tf_lite_model


def create_pruned_model(model, test_set):
    """
    Method to create and evaluate a pruned version of given `model`

    :param model: the model to prune
    :param test_set: test set for performance validation
    :return: pruned version of `model`
    """
    model_for_export = tfmot.sparsity.keras.strip_pruning(model)

    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()

    _, pruned_tflite_file = tempfile.mkstemp('.tflite')

    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)

    print('Saved pruned TFLite model to:', pruned_tflite_file)
    print('Model in Mb:', os.path.getsize(pruned_tflite_file) / float(2 ** 20))

    test_images, test_labels = tuple(zip(*test_set.unbatch()))
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    tflite_predictions = evaluate_q_model(pruned_tflite_model, test_images)
    model_prediction = model.predict(test_images, verbose=0)
    validate_q_model_prediction(model_prediction, tflite_predictions, test_labels, model.model_type)

    return pruned_tflite_model


def cluster_weights(model, cluster_method: ClusterMethod, number_clusters):
    """
    Clusters weights of given model in `number_clusters` clusters, using given method `cluster_method`.

    **Note this will change the underlying weights of `model`.
    In case you want to validate your model, perform a prediction prior to calling this method!**

    Hint:
    Use this in combination with CentroidInitialization.KMEANS_PLUS_PLUS
    for MobileNet on the Regression task, this preserves weights in the domain 1e-15 well.

    :param model:
    :param cluster_method: one of CentroidInitialization.KMEANS_PLUS_PLUS, CentroidInitialization.DENSITY_BASED, CentroidInitialization.RANDOM, CentroidInitialization.LINEAR
    :param number_clusters:
    :return:
    """
    clustering_params = {
        'number_of_clusters': number_clusters,
        'cluster_centroids_init': cluster_method.value
    }

    # Cluster a whole model
    clustered_model = tf_cluster_weights(model, **clustering_params)
    keras_model = tfmot.clustering.keras.strip_clustering(clustered_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    return converter.convert()
