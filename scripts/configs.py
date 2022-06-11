"""

"""
import functools
from typing import Dict, Tuple

from tensorflow.keras.applications.mobilenet import \
    preprocess_input as mobilenet_preprocess_input

from src.data.load_dataset import directory_to_classification_dataset
from src.losses.giou_loss import GIoULoss
from src.models.solvers.independent import IndependentModel
from src.models.solvers.two_in_one import TwoInOneTFLite
from src.models.solvers.two_stage import TwoStageModel


def create_conversion_config(input_shape: Tuple[int, int, int]) -> Dict:
    """
    Creates configuration to convert trained TF models to TFLite models

    :param input_shape: shape of the expected input features
    :return: Conversion configuration
    """
    from src.models.tf_lite.convert_to_tflite import (ClusterMethod,
                                                      QuantizationMethod,
                                                      cluster_weights,
                                                      create_quantize_model)
    cropped_test_set, cropped_train_set, _ = directory_to_classification_dataset(
        'data/iNat/cropped-data',
        img_width=input_shape[1],
        img_height=input_shape[0]
    )

    test_set, train_set, _ = directory_to_classification_dataset(
        'data/iNat/data',
        img_width=input_shape[1],
        img_height=input_shape[0]
    )

    train_set = train_set.map(lambda x, y: (mobilenet_preprocess_input(x), y))
    test_set = test_set.map(lambda x, y: (mobilenet_preprocess_input(x), y))
    cropped_train_set = cropped_train_set.map(lambda x, y: (mobilenet_preprocess_input(x), y))
    cropped_test_set = cropped_test_set.map(lambda x, y: (mobilenet_preprocess_input(x), y))

    return {
        'independent': {
            'model_cls': IndependentModel,
            'clf': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'classifier',
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/uncropped-clf-mobilenet/full.h5',
                'learning_rate': 0.0005,
            },
            'reg': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'regressor',
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/augmented-bbreg-mobilenet/full.h5',
                'loss': GIoULoss(),
                'learning_rate': 0.0005,
            },
            'is_tflite': True,
            'method': {
                'classifier': functools.partial(
                    create_quantize_model,
                    train_set=train_set,
                    test_set=test_set,
                    quant_method=QuantizationMethod.DYNAMIC
                ),
                'regressor': functools.partial(cluster_weights, cluster_method=ClusterMethod.KMEANS_PLUS_PLUS,
                                               number_clusters=16)
            }
        },
        'two-stage': {
            'model_cls': TwoStageModel,
            'clf': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'classifier',
                    'regularization_factor': 0.01,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/clf-mobilenet/full.h5',
                'learning_rate': 0.005,
            },
            'reg': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'regressor',
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/augmented-bbreg-mobilenet/full.h5',
                'loss': GIoULoss(),
                'learning_rate': 0.0005,
            },
            'is_tflite': True,
            'method': {
                'classifier': functools.partial(
                    create_quantize_model,
                    train_set=cropped_train_set,
                    test_set=cropped_test_set,
                    quant_method=QuantizationMethod.DYNAMIC
                ),
                'regressor': functools.partial(cluster_weights, cluster_method=ClusterMethod.KMEANS_PLUS_PLUS,
                                               number_clusters=16)
            }
        },
        'single-stage': {
            'model_cls': TwoInOneTFLite,
            'head': {
                'head': {
                    'dense_neurons': 128,
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.6068936417692536,
                    'batch_size': 32,
                    'num_classes': 5,
                },
                'weights': 'weights/augmented-2-in-1-mobilenet/full.h5',
                'learning_rate': 0.0004503371828412395,
            },
            'is_tflite': True,
            'method': {
                'all': functools.partial(cluster_weights, cluster_method=ClusterMethod.KMEANS_PLUS_PLUS,
                                         number_clusters=16),
            }
        },
    }


def create_config():
    """
    Creates configuration to build three final models:

    #. Independent

    #. Two-Stage

    #. Single-Stage

    :return: Final model configuration
    """
    return {
        'independent': {
            'model_cls': IndependentModel,
            'clf': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'classifier',
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/uncropped-clf-mobilenet/full.h5',
                'learning_rate': 0.0005,
            },
            'reg': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'regressor',
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/augmented-bbreg-mobilenet/full.h5',
                'loss': GIoULoss(),
                'learning_rate': 0.0005,
            },
            'is_tflite': True,
        },
        'two-stage': {
            'model_cls': TwoStageModel,
            'clf': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'classifier',
                    'regularization_factor': 0.01,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/clf-mobilenet/full.h5',
                'learning_rate': 0.005,
            },
            'reg': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'regressor',
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/augmented-bbreg-mobilenet/full.h5',
                'loss': GIoULoss(),
                'learning_rate': 0.0005,
            },
            'is_tflite': True,
        },
        'single-stage': {
            'model_cls': TwoInOneTFLite,
            'head': {
                'head': {
                    'dense_neurons': 128,
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.6068936417692536,
                    'batch_size': 32,
                    'num_classes': 5,
                },
                'weights': 'weights/augmented-2-in-1-mobilenet/full.h5',
                'learning_rate': 0.0004503371828412395,
            },
            'is_tflite': True,
        },
    }


def create_tflite_config():
    """
    Creates TFLite configuration for the final three models:

    #. Independent

    #. Two-Stage

    #. Single-Stage

    :return: Final TFLite model configuration
    """
    return {
        'independent': {
            'model_cls': IndependentModel,
            'clf': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'classifier',
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/independent-classifier.tflite',
                'learning_rate': 0.0005,
            },
            'reg': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'regressor',
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/independent-regressor.tflite',
                'loss': GIoULoss(),
                'learning_rate': 0.0005,
            },
            'is_tflite': True
        },
        'two-stage': {
            'model_cls': TwoStageModel,
            'clf': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'classifier',
                    'regularization_factor': 0.01,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/two-stage-classifier.tflite',
                'learning_rate': 0.005,
            },
            'reg': {
                'head': {
                    'dense_neurons': 128,
                    'name': 'regressor',
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.5,
                    'batch_size': 32,
                    'include_pooling': True
                },
                'weights': 'weights/two-stage-regressor.tflite',
                'loss': GIoULoss(),
                'learning_rate': 0.0005,
            },
            'is_tflite': True,
        },
        'single-stage': {
            'model_cls': TwoInOneTFLite,
            'head': {
                'head': {
                    'dense_neurons': 128,
                    'regularization_factor': 0.0001,
                    'dropout_factor': 0.6068936417692536,
                    'batch_size': 32,
                    'num_classes': 5,
                },
                'weights': 'weights/single-stage-all.tflite',
                'learning_rate': 0.0004503371828412395,
            },
            'is_tflite': True
        },
    }
