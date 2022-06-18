import base64
import io
import os
import re
from typing import Dict, List

import numpy as np
from PIL import Image


def get_example_files() -> List[Dict]:
    file_names = os.listdir('./val-data')
    files = []
    for name in file_names:
        with open(f'./val-data/{name}', 'rb') as image_file:
            files.append({'src': base64.b64encode(image_file.read()).decode('utf-8'), 'name': name})
    return files


def load_example_file(file_name):
    return np.array(Image.open(f'./val-data/{file_name}').resize((224, 224)))


def decode_img(msg):
    """
    decode base64 encoded image

    :param msg:
    :return:
    """
    msg = re.sub('^data:image/.+;base64,', '', msg)
    msg = base64.b64decode(bytes(msg, 'utf-8'))
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    return img
