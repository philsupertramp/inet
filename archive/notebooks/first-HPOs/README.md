# Benchmarks
The following document contains logs of Benchmarks, for different architectures
### Used layer names and descriptions
- BN: Batch Normalization
- ReLU: Activation function
- Conv (`n x (w x h)`): Convolutional Layer (`n`: Number of filter masks, `w`: Filter mask width, `h`: Filter mask height)
- Dense (`n`): Dense Layer (`n`: Number of Neurons)
- ConvBlock (`n x (w x h)`):
  - Conv (`n x (w x h)`)
  - BN
  - ReLU
- DenseBlock (`n`):
  - BN
  - ReLU
  - Dense (`n`)

## Bounding Box Regression
Objective: minimize `val_root_mean_squared_error`

| Architecture  | Loss  | GIoU        | RMSE        | Learning Rate          | Runtime  | Number Parameters |
|:-------------:|:------|:------------|:------------|:-----------------------|:---------|:------------------|
|    VGG-16     | `mse` | `1.579256`  | `44.000324` | `0.01`                 | `~3:30h` | `14,714,688`      |
|   MobileNet   | `mse` | `0.6044541` | `16.54146`  | `0.01`                 | `~3:30h` | `3,294,982`       |
| Custom-CNN v1 | `mse` | `0.6690553` | `18.344604` | `0.01`                 | `~1:35h` | `411,570`         |
| Custom-CNN v2 | `mse` | `0.6599949` | `18.533228` | `0.005526762501321664` | `~4:20h` | `412,546`         |
|  Const. Reg   | `–`   | `0.6561564` | `21.484241` | –                      | `~1 min` | train sample size |
|   Ridge Reg   | `–`   | `0.6492654` | `20.932585` | –                      | `~1 min` | `100`             |
|   Ridge Reg   | `–`   | `0.6474993` | `20.932585` | –                      | –        | `400`             |

### Architectures
- Custom-CNN v1 (`alpha: 0.0222749513617932`)
  - Backbone
    - BN
    - ConvBlock (16x(3x3))
    - ConvBlock (32x(3x3))
    - ConvBlock (64x(3x3))
    - ConvBlock (128x(3x3))
    - ConvBlock (256x(3x3))
    - GlobalMaxPooling
  - Task solver
    - Dropout (0.75)
    - Dense (64)
    - DenseBlock (4)
- Custom-CNN v2 (`alpha: 0.0020237367829352643`)
  - Backbone
    - BN
    - ConvBlock (8x(3x3))
    - ConvBlock (16x(3x3))
    - ConvBlock (32x(3x3))
    - ConvBlock (64x(3x3))
    - ConvBlock (128x(3x3))
    - ConvBlock (256x(3x3))
    - GlobalMaxPooling
  - Task solver
    - Dropout (0.36554832151542094)
    - Dense (64)
    - DenseBlock (4)


### HPO Parameters
- BBReg:
  - dense_neurons: Number neurons in Task Dense layer
  - alpha: regularization parameter for Task DenseBlock
  - dropout: dropout factor used in Task Model
  - batch_size: Batch size to train on
  - learning_rate: Learning Rate to start with
  - loss: Loss used for training

- BBReg-CNN-Backbone:
  - Parameters from BBReg-CNN
  - num_layers: Number ConvBlocks for the model
  - filter_size_start: Number Filter masks to start with in first ConvBlock, all consecutive have `n_i = (n_{i - 1}) * 2` masks


## Classification
Objective: maximize `val_accuracy`

| Architecture  | Accuracy | f1                  | Learning Rate | Batch size | Runtime | Number Parameters |
|---------------|----------|---------------------|---------------|------------|---------|-------------------|
| Custom-CNN v3 | `0.4422` | `0.430622175108589` | `0.01`        | `64`       | `~2h`   | `462,705`         |
| Custom-CNN v4 | `0.6755` | `0.674135469679054` | `0.0001`      | `16`       | `3:45h` | `25,690,001`      |
| VGG-16        | `0.2089` | `0.069117647058823` | `0.01`        | `64`       | `~4h`   | `14,714,688`      |
| MobileNet     | `0.4333` | `0.449116649066226` | `0.01`        | `64`       | `~4h`   | `3,294,982`       |
| SVM (C=0.1)   | `0.4828` | `0.474813648913584` | `–`           | `–`        | `~1min` | `100`             |
| SVM (C=0.1)   | `0.5339` | `0.520047289383970` | `–`           | `–`        | `~1min` | `400`             |

### Architectures
- Custom-CNN v3 (`alpha: 0.0222749513617932`)
  - Backbone
    - BN
    - ConvBlock (16x(3x3))
    - ConvBlock (32x(3x3))
    - ConvBlock (64x(3x3))
    - ConvBlock (128x(3x3))
    - ConvBlock (256x(3x3))
    - GlobalMaxPooling
  - Task solver
    - Dropout (0.75)
    - Dense (256)
    - DenseBlock (5)
    - random weight initialization
- Custom-CNN v4 (`alpha: 0.020964029604508203`)
  - Backbone
    - BN
    - ConvBlock (64x(3x3))
    - ConvBlock (128x(3x3))
    - ConvBlock (256x(3x3))
    - ConvBlock (512x(3x3))
    - ConvBlock (1024x(3x3))
    - ConvBlock (2048x(3x3))
    - GlobalMaxPooling
  - Task solver
    - Dropout (0.1)
    - Dense (256)
    - DenseBlock (5)
    - random weight initialization


## HPO Parameters:
- learning_rate: learning rate to use for training
- batch_size: batch size to train with
- dense_neurons: number neurons in dense layer before output
- alpha: regularization parameter for output layer
- dropout: dropout factor for task
- frozen_blocks: `0` all trainable, `1` 50% trainable, `2` 0% trainable backbone blocks
- load_weights: boolean to load pretrained weights from Custom-CNN v1
