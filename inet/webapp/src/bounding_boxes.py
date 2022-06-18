import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from inet.models.data_structures import BoundingBox


def build_bb_image(image, bb):
    plt.figure()
    plt.imshow(image)
    bb.plot()
    plt.axis('off')
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode('utf-8')


def make_prediction(image, regressor):
    inpt = regressor.preprocess_input([image])
    prediction = regressor.predict(inpt)[0].reshape((2, 2))
    prediction[:, 0] *= image.shape[0]
    prediction[:, 1] *= image.shape[1]
    bb = BoundingBox(*prediction.reshape(4))
    return build_bb_image(image, bb)
