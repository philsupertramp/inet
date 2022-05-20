import os
import time
from typing import Optional, Tuple

from tensorflow import keras

cur_dir = os.path.dirname(__file__)
train_logdir = os.path.join(cur_dir, '../train_logs')


def get_train_logdir(name: Optional[str] = None):
    if name is None:
        name = ''
    run_id = time.strftime(f'{name}-run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(train_logdir, run_id)


def copy_model(input_model, layer_range: Tuple[int, int] = (0, -1)):
    return keras.models.Model(inputs=input_model.input, outputs=input_model.layers[layer_range[1]].output)
