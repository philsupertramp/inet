"""
Inference tests using TFLite models
"""
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import \
    preprocess_input as mobilenet_preprocess_input

from scripts.configs import create_tflite_config
from src.data.load_dataset import directory_to_two_in_one_dataset
from tests.helper import Timer

if __name__ == '__main__':
    input_shape = (224, 224, 3)
    model_configs = create_tflite_config()
    results = {}

    for name, config in model_configs.items():
        test_set, _, _ = directory_to_two_in_one_dataset(
            'data/iNat/data',
            img_width=input_shape[1],
            img_height=input_shape[0]
        )
        mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        if 'reg' in config:
            config['reg']['backbone'] = mobilenet
        if 'clf' in config:
            config['clf']['backbone'] = mobilenet
        if 'head' in config:
            config['head']['backbone'] = mobilenet
        model_cls = config.get('model_cls')
        model = model_cls.from_config(config, is_tflite=True)

        print(f'Model evaluation for: "{name}":')
        model.evaluate_model(test_set, mobilenet_preprocess_input, True)

        print(f'Testing inference for 5 samples for: "{name}":')
        timer = Timer()
        unbatched_test_set = test_set.unbatch().batch(1)
        for _ in range(5):
            sample = unbatched_test_set.take(1)
            sample_value, _ = tuple(zip(*sample.unbatch()))
            sample_value = np.array(sample_value)
            with timer:
                model.predict(sample_value)

        seconds = sum(i.microseconds/1_000_000 for i in timer.results)
        results[name] = {
            'seconds': seconds,
            'all': timer.results
        }
        print(f'AVG inference time: {seconds}s.')

    print(results)
