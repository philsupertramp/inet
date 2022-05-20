from datetime import datetime

import tensorflow as tf

from src.models.tflite_methods import evaluate_interpreted_model


class Timer:
    def __init__(self):
        self.results = []
        self.current_start = None

    def __enter__(self):
        self.current_start = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.results.append(datetime.now() - self.current_start)


def build_tf_model_from_file(filename):
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter


def evaluate_q_model_from_file(tf_lite_model, test_images):
    interpreter = tf.lite.Interpreter(model_path=tf_lite_model)
    interpreter.allocate_tensors()

    return evaluate_interpreted_model(interpreter, test_images)
