from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import \
    preprocess_input as mobilenet_preprocess_input

from src.data.load_dataset import directory_to_classification_dataset
from src.losses.giou_loss import GIoULoss
from src.models.two_stage import TwoStageModel

if __name__ == '__main__':
    input_shape = (224, 224, 3)
    _, _, validation_set = directory_to_classification_dataset('data/iNat/data', img_width=input_shape[1],
                                                               img_height=input_shape[0])
    mobilenet = MobileNet(weights=None, include_top=False, input_shape=input_shape)

    two_in_one_architecture = {
        'reg': {
            'backbone': mobilenet,
            'head': {
                'dense_neurons': 128,
                'name': 'regressor',
                'regularization_factor': 0.0001,
                'dropout_factor': 0.5,
                'batch_size': 32,
                'include_pooling': True
            },
            'weights': 'model-weights/augmented-bbreg-mobilenet/full.h5',
            'loss': GIoULoss(),
            'learning_rate': 0.0005,
            'is_tflite': True
        },
        'clf': {
            'backbone': mobilenet,
            'head': {
                'dense_neurons': 128,
                'name': 'classifier',
                'regularization_factor': 0.01,
                'dropout_factor': 0.5,
                'batch_size': 32,
                'include_pooling': True
            },
            'weights': 'model-weights/clf-mobilenet/full.h5',
            'loss': 'categorical_crossentropy',
            'learning_rate': 0.005,
            'is_tflite': True
        },
    }
    two_stage_model = TwoStageModel.from_config(two_in_one_architecture)

    two_stage_model.evaluate_model(
        validation_set,
        mobilenet_preprocess_input,
        render_samples=True
    )
