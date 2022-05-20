# HPO for Model Architectures
Winning configurations per task are marked **bold**.
All networks consist of
1. A Backbone Model for feature extraction
2. A Task solving Head for Bounding Box Regression or Classification respectively

`PNet` is the _Work in Progress_ name of a CNN Architecture:
- BN
- ConvBlock (16x(3x3))
- ConvBlock (32x(3x3))
- ConvBlock (64x(3x3))
- ConvBlock (128x(3x3))
- ConvBlock (256x(3x3))

ConvBlock:
- Batch-Normalization
- ReLU
- Convolutional Layer

## HPO Parameters:
- Learning Rate
- Number Dense Neurons
- Regularization Factor in Dense Block

# Regression
## Conventional Methods
|   Method   |   GIoU    |   RMSE    |
|:----------:|:---------:|:---------:|
| Average BB | 0.6561564 | 21.484241 |
| Ridge Reg  | 0.6492654 | 20.932585 |
| Ridge Reg  | 0.6474993 | 20.932585 |


## CNNS
Head:
- GlobalMaxPooling
- Dense Layer (# neurons part of HPO)
- DenseBlock
  - Batch-Normalization
  - ReLU
  - Dense Layer (4 Neurons, ReLU activation)


## Unaugmented (`Regression-HPO.ipynb`):
| Backbone Model |   GIoU    |   RMSE    |   LR   |  HPO Time   |
|:--------------:|:---------:|:---------:|:------:|:-----------:|
|      PNet      | 0.6160481 | 24.948946 |  0.01  |     ~6h     |
|   MobileNet    | 0.6154345 | 25.961775 |  0.01  |     ~8h     |
|     VGG-16     | 1.579256  | 44.000324 | 0.0001 | 09h 57m 32s |

## Augmented (`Augmented-Regression-HPO.ipynb`):
| Backbone Model |     GIoU      |     RMSE      |    LR     |    HPO Time     |
|:--------------:|:-------------:|:-------------:|:---------:|:---------------:|
|      PNet      |   0.6215299   |   28.707605   |  0.0001   |   11h 15m 26s   |
| **MobileNet**  | **0.5970283** | **24.924677** | **0.005** | **19h 45m 57s** |
|     VGG-16     |   1.1301343   |   45.17516    |   0.001   |   22h 02m 08s   |


# Classification
## Conventional Methods
|   Method    | Accuracy  |        F1         |
|:-----------:|:---------:|:-----------------:|
| SVM (C=0.1) |  0.4828   | 0.474813648913584 |
| SVM (C=0.1) |  0.5339   | 0.520047289383970 |

## CNNs
Head:
- GlobalMaxPooling
- Dense Layer (# neurons part of HPO)
- DenseBlock
  - Batch-Normalization
  - ReLU
  - Dense Layer (5 Neurons, softmax activation)


## Unaugmented (`Classification-HPO.ipynb`):
| Backbone Model | Accuracy  |        F1         |  LR   |  HPO Time   |
|:--------------:|:---------:|:-----------------:|:-----:|:-----------:|
|      PNet      | 0.4355555 |     0.4230925     | 0.01  | 01h 40m 20s |
|   MobileNet    | 0.8555555 |     0.8512115     | 0.005 | 03h 44m 09s |
|     VGG-16     | 0.8577777 |     0.8562220     | 0.01  | 05h 13m 51s |


### Using Architecture as suggested by keras-docs (`Classification-HPO.ipynb` bottom)
[Keras-Docs](https://keras.io/api/applications/#usage-examples-for-image-classification-models)

| Backbone Model |   Accuracy    |      F1       |    LR     |  HPO Time   |
|:--------------:|:-------------:|:-------------:|:---------:|:-----------:|
| **MobileNet**  | **0.8866666** | **0.8839064** | **0.01**  |   **~4h**   |
|     VGG-16     |   0.8755555   |   0.8748135   |  0.0001   |   ~4h 30m   |

### Pretrained Weights from Regression Task (`Transfer-Classification-HPO.ipynb`):
| Backbone Model | Accuracy  |    F1     |  LR  | Retrained Layers |       HPO Time        |
|:--------------:|:---------:|:---------:|:----:|:----------------:|:---------------------:|
|      PNet      | 0.5022222 | 0.4899976 | 0.01 |       all        |      01h 40m 39s      |
|   MobileNet    | 0.4911111 | 0.4869583 | 0.01 |       all        |      03h 30m 09s      |
|     VGG-16     | 0.2088888 | 0.0691176 | 0.01 |       all        |      05h 20m 16s      |

## Augmented (`Augmented-Classification-HPO.ipynb`):
| Backbone Model | Accuracy  |    F1     |   LR   |  HPO Time   |
|:--------------:|:---------:|:---------:|:------:|:-----------:|
|      PNet      | 0.6177777 | 0.6045950 | 0.0001 | 15h 46m 39s |
|   MobileNet    | 0.8355555 | 0.8296459 | 0.005  | 11h 16m 10s |
|     VGG-16     | 0.8333333 | 0.8282507 | 0.005  | 16h 59m 16s |
