import tempfile

import numpy as np
import sklearn
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError

from scripts.helpers import ProgressBar
from src.losses.giou_loss import GIoULoss
from src.models.constants import ModelType


def save_model_file(model):
    _, keras_file = tempfile.mkstemp('.h5')
    model.save(keras_file, include_optimizer=False)
    return keras_file


def get_gzipped_model_size(model):
    # It returns the size of the gzipped model in bytes.
    import os
    import zipfile

    keras_file = save_model_file(model)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)
    return os.path.getsize(zipped_file)


def evaluate_interpreted_model(interpreter, test_images):
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
    interpreter = tf.lite.Interpreter(model_content=tf_lite_model)
    interpreter.allocate_tensors()

    return evaluate_interpreted_model(interpreter, test_images)


def evaluate_regression(model_predictions, tfl_model_predictions, test_labels):
    gloss_fn = GIoULoss()
    rmse_fn = RootMeanSquaredError()

    def model_eval(pred, name):
        print(f'"{name}" GIoU:', 1. - gloss_fn(test_labels, pred).numpy())
        print(f'"{name}" RMSE:', rmse_fn(test_labels, pred).numpy())

    model_eval(model_predictions, 'Original')
    model_eval(tfl_model_predictions, 'TFLite')


def evaluate_classification(model_predictions, tfl_model_predictions, test_labels):
    def model_eval(pred, name):
        print(f'"{name}" Accuracy:', sklearn.metrics.accuracy_score(test_labels, pred, normalize=True))
        print(f'"{name}" F1-Score:', sklearn.metrics.f1_score(test_labels, pred, average='macro'))

    model_eval(model_predictions, 'Original')
    model_eval(tfl_model_predictions, 'TFLite')


def evaluate_two_in_one(model_predictions, tfl_model_predictions, test_labels):
    gloss_fn = GIoULoss()
    rmse_fn = RootMeanSquaredError()

    def model_eval(pred, name):
        print(f'"{name}" Accuracy:', sklearn.metrics.accuracy_score(test_labels[:, 0], pred[:, 0], normalize=True))
        print(f'"{name}" F1-Score:', sklearn.metrics.f1_score(test_labels[:, 0], pred[:, 0], average='macro'))
        print(f'"{name}" GIoU:', 1. - gloss_fn(test_labels[:, 1], pred[:, 1]).numpy())
        print(f'"{name}" RMSE:', rmse_fn(test_labels[:, 1], pred[:, 1]).numpy())

    model_eval(model_predictions, 'Original')
    model_eval(tfl_model_predictions, 'TFLite')


def validate_q_model_prediction(model_prediction, tfl_model_prediction, test_labels, model_type):
    if model_type == ModelType.CLASSIFICATION.value:
        evaluate_classification(model_prediction.argmax(axis=1), np.array(tfl_model_prediction).argmax(axis=2).flatten(), test_labels.argmax(axis=1))
    elif model_type == ModelType.REGRESSION.value:
        evaluate_regression(model_prediction, tfl_model_prediction, test_labels)
    elif model_type == ModelType.TWO_IN_ONE.value:
        evaluate_two_in_one(model_prediction, tfl_model_prediction, test_labels)
