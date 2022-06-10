import json
import os
import re
import time
from enum import Enum
from typing import List

import matplotlib.pyplot as plt


class FrozenBlockConf(Enum):
    """Helper to freeze layers in a sequential model"""
    ## Train all layers
    TRAIN_ALL = 0
    ## Train 50% of the layers
    TRAIN_HALF = 1
    ## Train 0 layers
    TRAIN_NONE = 2

    def process(self, backbone) -> None:
        """
        Method to propagate selection through a model instance
        :param backbone: the model to configure
        :return:
        """
        if self == FrozenBlockConf.TRAIN_NONE:
            backbone.trainable = False
            for layer in backbone.layers:
                layer.trainable = False
        elif self == FrozenBlockConf.TRAIN_ALL:
            backbone.trainable = True
            for layer in backbone.layers:
                layer.trainable = True
        elif self == FrozenBlockConf.TRAIN_HALF:
            layer_names = list(filter(lambda x: 'conv' in x, [layer.name for layer in backbone.layers]))
            unique_ids = list({int(re.findall(r'\d', i)[0]) for i in layer_names})
            last_frozen_ids = max(unique_ids[:(len(unique_ids)//2)+1])
            for layer in backbone.layers[:last_frozen_ids]:
                layer.trainable = True
            for layer in backbone.layers[last_frozen_ids:]:
                layer.trainable = False

    @staticmethod
    def choices():
        """Helper to get list of available choices"""
        return [FrozenBlockConf.TRAIN_NONE.value, FrozenBlockConf.TRAIN_HALF.value, FrozenBlockConf.TRAIN_ALL.value]


def read_trials(dir_name: str) -> List:
    """Reads trial files provided by [keras-tuner](https://keras.io/keras_tuner/)."""
    trial_list = [os.path.join(directory, 'trial.json') for directory in filter(
        lambda x: os.path.isdir(x),
        [os.path.join(dir_name, f) for f in os.listdir(dir_name) if 'trial' in f]
    )]

    # order by creation timestamp, considered order
    trial_list.sort(key=lambda x: time.ctime(os.path.getctime(x)))

    trials = list()
    for trial in trial_list:
        with open(trial) as f:
            trials.append(json.load(f))

    return trials


def plot_hpo_values(trial) -> None:
    """Helper to display course of HP values during a HPO"""
    hp_list = [t.get('hyperparameters').get('values') for t in trial]
    hps = {}
    for elem in hp_list:
        for key, value in elem.items():
            if key in hps:
                hps[key].append(value)
            else:
                hps[key] = [value]

    for key, values in hps.items():
        plt.title(f'Chosen values for HP "{key}"')
        plt.plot(list(range(len(values))), values, '*-')
        plt.show()
