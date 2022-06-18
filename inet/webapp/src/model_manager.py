import json
import os
from typing import Optional

from inet.models.architectures.bounding_boxes import BoundingBoxRegressor


class ModelManager:
    """
    Singleton implementation for model management
    use `model_manager.model_manager` in code.
    """
    ## Source directory of model weights
    MODEL_DIR = 'model-weights'
    ## Model Name-Constructor map
    MODEL_MAP = {
        'BoundingBoxRegressor': BoundingBoxRegressor
    }

    def __init__(self, default_model: Optional = None):
        """

        :param default_model: Optional defaMulti processing implementation of files movingult model
        """
        ## Storage of available models
        self.models = {'default': default_model}
        ## The default model to use
        self.default_model = default_model

    def load_models(self):
        """
        Loads models from MODEL_DIR directory
        :return:
        """
        files = set(os.listdir(self.MODEL_DIR))
        assert 'content.json' in files, 'No content file, cannot load pretrained models!'

        files -= {'content.json'}
        with open(os.path.join(self.MODEL_DIR, 'content.json')) as file:
            configs = json.load(file)

        for config in configs:
            file = config['weights']
            model_cls = self.MODEL_MAP.get(config['model'])
            model = model_cls(model_name=file.split('.')[0], **config['parameters'])
            model.load_weights(os.path.join(self.MODEL_DIR, file))
            model.trainable = False
            self.models[model.model_name] = model
            if not self.models.get('default'):
                self.models['default'] = model

    def select(self, name):
        """Select a model by name as current default"""
        self.models['default'] = self.models.get(name, self.default_model)
        self.default_model = self.models['default']

    def get(self, name: str = 'default'):
        """Model getter by name, omit name for default model"""
        return self.models.get(name, self.default_model)

    def get_all_models(self):
        """Getter for all available model names"""
        return self.models.keys()


## Singleton instance of `ModelManager`
model_manager = ModelManager()
