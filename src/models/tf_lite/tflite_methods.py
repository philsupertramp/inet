import tempfile

import numpy as np
import sklearn
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError

from scripts.helpers import ProgressBar
from src.data.constants import ModelType
from src.losses.giou_loss import GIoULoss


def save_model_file(model):
    """Writes model to .h5 file"""
    _, keras_file = tempfile.mkstemp('.h5')
    model.save(keras_file, include_optimizer=False)
    return keras_file


def get_gzipped_model_size(model):
    """Computes size of gzip converted model"""
    # It returns the size of the gzipped model in bytes.
    import os
    import zipfile

    keras_file = save_model_file(model)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)
    return os.path.getsize(zipped_file)


def evaluate_interpreted_model(interpreter, test_images):
    """
    Method to evaluate an interpreted (tflite) model

    :param interpreter: interpreted model
    :param test_images: input to evaluate
    :return: predictions of interpreted model
    """
    input_index = interpreter.get_input_details()[0]['index']
    outputs_indices = [o['index'] for o in interpreter.get_output_details()]

    # Run predictions on every image in the "test" dataset.
    predictions = []
    set_len = len(test_images)
    pb = ProgressBar(set_len)
    for i, test_image in enumerate(test_images):
        pb.step(i)
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension
        current_outputs = []
        for output_index in outputs_indices:
            current_outputs.append(interpreter.get_tensor(output_index))

        if len(current_outputs) == 1:
            predictions.append(current_outputs[0])
        else:
            predictions.append(current_outputs)

    pb.done()
    return predictions


def evaluate_q_model(tf_lite_model, test_images):
    """
    Evaluation method for quantization aware model

    :param tf_lite_model: q-aware tflite model
    :param test_images: input to perform prediction on
    :return: predictions for given images
    """
    interpreter = tf.lite.Interpreter(model_content=tf_lite_model)
    interpreter.allocate_tensors()

    return evaluate_interpreted_model(interpreter, test_images)


def evaluate_regression(model_predictions, tfl_model_predictions, test_labels):
    """
    Evaluation method for TFLite regression model

    :param model_predictions: predictions done by the original model
    :param tfl_model_predictions: predictions done by the tflite version of the original model
    :param test_labels: ground truth labels
    :return:
    """
    gloss_fn = GIoULoss()
    rmse_fn = RootMeanSquaredError()

    def model_eval(pred, name):
        print(f'"{name}" GIoU:', 1. - gloss_fn(test_labels, pred).numpy())
        print(f'"{name}" RMSE:', rmse_fn(test_labels, pred).numpy())

    model_eval(model_predictions, 'Original')
    model_eval(tfl_model_predictions, 'TFLite')


def evaluate_classification(model_predictions, tfl_model_predictions, test_labels):
    """
    Evaluation of classification model

    :param model_predictions: predictions done by the original model
    :param tfl_model_predictions: predictions done by the tflite version of the original model
    :param test_labels: ground truth labels
    :return:
    """
    def model_eval(pred, name):
        print(f'"{name}" Accuracy:', sklearn.metrics.accuracy_score(test_labels, pred, normalize=True))
        print(f'"{name}" F1-Score:', sklearn.metrics.f1_score(test_labels, pred, average='macro'))

    model_eval(model_predictions, 'Original')
    model_eval(tfl_model_predictions, 'TFLite')


def evaluate_two_in_one(model_predictions, tfl_model_predictions, test_labels):
    """
    Evaluation of two-in-one model

    :param model_predictions: predictions done by the original model
    :param tfl_model_predictions: predictions done by the tflite version of the original model
    :param test_labels: ground truth labels
    :return:
    """
    gloss_fn = GIoULoss()
    rmse_fn = RootMeanSquaredError()

    def model_eval(pred, name):
        print(f'"{name}" Accuracy:', sklearn.metrics.accuracy_score(test_labels[:, 0], pred[:, 0], normalize=True))
        print(f'"{name}" F1-Score:', sklearn.metrics.f1_score(test_labels[:, 0], pred[:, 0], average='macro'))
        print(f'"{name}" GIoU:', 1. - gloss_fn(test_labels[:, 1], pred[:, 1]).numpy())
        print(f'"{name}" RMSE:', rmse_fn(test_labels[:, 1], pred[:, 1]).numpy())

    model_eval(model_predictions, 'Original')
    model_eval(tfl_model_predictions, 'TFLite')


def validate_q_model_prediction(model_prediction, tfl_model_prediction, test_labels, model_type) -> None:
    """
    Validates a tflite model, comparing values with its original predecessor.

    :param model_prediction: predictions done by the original model
    :param tfl_model_prediction: predictions done by the tflite version of the original model
    :param test_labels: ground truth labels
    :param model_type: `ModelType` of the underlying model
    :return:
    """
    if model_type == ModelType.CLASSIFICATION.value:
        evaluate_classification(model_prediction.argmax(axis=1), np.array(tfl_model_prediction).argmax(axis=2).flatten(), test_labels.argmax(axis=1))
    elif model_type == ModelType.REGRESSION.value:
        evaluate_regression(model_prediction, tfl_model_prediction, test_labels)
    elif model_type == ModelType.TWO_IN_ONE.value:
        evaluate_two_in_one(model_prediction, tfl_model_prediction, test_labels)
