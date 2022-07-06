from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf

from inet.data.visualization import (plot_confusion_matrix,
                                     plot_prediction_samples)
from inet.losses.giou_loss import GIoULoss
from scripts.constants import CLASS_MAP


def evaluate_solver_predictions(
        predicted_labels, predicted_bbs, validation_values,
        validation_labels, render_samples, model_name: Optional[str] = None):
    """
    Common evaluation method for solvers

    Computes:
    - GIoU-Loss
    - RMSE
    - Accuracy
    - F1-Score

    :param predicted_labels: predicted class labels by the solver
    :param predicted_bbs:  predicted bboxes by the solver
    :param validation_values: input samples used to perform predictions
    :param validation_labels: ground truth labels for given samples
    :param render_samples: if `True` renders confusion matrix of classification and up to 25 samples of bbox regression
    :param model_name: used to save resulting plots
    :return:
    """
    predicted_labels_argmax = predicted_labels.copy()
    predicted_labels_argmax = predicted_labels_argmax.argmax(axis=1)
    labels = np.array([tf.argmax(i) for i in validation_labels[:, 0]])
    bbs = tf.constant([i.numpy() for i in validation_labels[:, 1]])

    acc = sklearn.metrics.accuracy_score(labels, predicted_labels_argmax)
    f1 = sklearn.metrics.f1_score(labels, predicted_labels_argmax, average='macro')
    giou = GIoULoss()(bbs, predicted_bbs).numpy()
    rmse = tf.keras.metrics.RootMeanSquaredError()(bbs, predicted_bbs).numpy()

    print(
        'Classification:\n',
        '=' * 35,
        '\n\tAccuracy:\t', acc,
        '\n\tf1 score:\t', f1,
        '\nLocalization:\n',
        '=' * 35,
        '\n\tGIoU:\t', 1. - giou,
        '\n\tRMSE:\t', rmse
    )
    if render_samples:
        plot_confusion_matrix(predicted_labels_argmax, labels, list(CLASS_MAP.keys()), normalize=True)
        plt.savefig(f'{model_name}-confusion.eps', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{model_name}-confusion.png', bbox_inches='tight', pad_inches=0)
        plot_prediction_samples(predicted_bbs, predicted_labels, validation_values)
        plt.savefig(f'{model_name}-predictions.eps', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{model_name}-predictions.png', bbox_inches='tight', pad_inches=0)
