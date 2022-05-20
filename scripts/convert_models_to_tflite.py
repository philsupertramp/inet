import os

import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet

from scripts.configs import create_conversion_config

if __name__ == '__main__':
    input_shape = (224, 224, 3)

    model_configs = create_conversion_config(input_shape)

    for name, config in model_configs.items():
        mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        if 'reg' in config:
            config['reg']['backbone'] = mobilenet
        if 'clf' in config:
            config['clf']['backbone'] = mobilenet
        if 'head' in config:
            config['head']['backbone'] = mobilenet
        model_cls = config.get('model_cls')
        model = model_cls.from_config(config)

        tf_lite_methods = config.get('method')
        for modelname, method in tf_lite_methods.items():
            if modelname == 'regressor':
                sub = model.regressor
            elif modelname == 'classifier':
                sub = model.classifier
            elif modelname == 'all':
                sub = model
            sub = method(sub)
            tf_file = f'../weights/{name}-{modelname}.tflite'
            # Save the TFLite model
            with tf.io.gfile.GFile(tf_file, 'wb') as f:
                f.write(sub)

            print('Model in Mb:', os.path.getsize(tf_file) / float(2 ** 20))
