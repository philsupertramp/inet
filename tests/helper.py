from datetime import datetime

import tensorflow as tf

from inet.models.tf_lite.tflite_methods import evaluate_interpreted_model


class Timer:
    """
    Helper class to measure inference time.

    Example:
        >>> import time
        >>> timer = Timer()
        >>> with timer:
        >>>     time.sleep(1)
        >>> print(timer.results[0])
    """
    def __init__(self):
        """
        Creates empty timer object
        """
        ## storage for timer results
        self.results = []
        ## determines start of a run
        self.current_start = None

    def __enter__(self):
        """
        Starts the timer
        """
        self.current_start = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stops the timer (and resets its start time)
        """
        self.results.append(datetime.now() - self.current_start)
        self.current_start = None


def build_tf_model_from_file(filename):
    """
    Method to create tflite model from file

    :param filename: tflite model file
    :return: interpreted tflite model
    """
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter


def evaluate_q_model_from_file(tf_lite_model, test_images):
    """
    Creates and evaluates a tflite model

    :param tf_lite_model: a tflite model file
    :param test_images: input to evaluate
    :return: prediction of the model
    """
    return evaluate_interpreted_model(build_tf_model_from_file(tf_lite_model), test_images)
