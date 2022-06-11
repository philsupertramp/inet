### TOC
- [Metrics](#metrics)
- [Regression](#regression-reg)
    - [Default Dataset](#default-dataset-reg1)
    - [Augmented Dataset](#augmented-dataset-reg2)
- [Classification](#classification-clf)
  - [Default Dataset](#default-dataset-clf1)
  - [Augmented Dataset](#augmented-dataset-clf2)
  - [Uncropped Dataset](#uncropped-dataset-clf3)
- [2-Head-Predictor](#2-head-predictor-2head) 
  - [Default Dataset](#default-dataset-2head1) 
  - [Augmented Dataset](#augmented-dataset-2head2)
- [MobileNet-SSD](#yolomobilenet-ssd-ssd)
- [Comparisons](#comparisons)

.. _metrics:
### Metrics:
.. _regression_metrics:
##### Regression:
.. _giou:
###### GIoU
During training and in log entry you can find `GIoULoss = 1 - GIoU` as **GIoU Loss**.
The metric used for comparison is GIoU: `1 - GIoULoss = GIoU`

.. _rmse:
###### RMSE
RMSE same value as during training, except variance due to dropout.

.. _classification_metrics:
##### Classification:
.. _accuracy:
###### Accuracy:
Regular accuracy: `ACC = (TP + TN)/(TP + TN + FP + FN)`

.. _f1:
###### F1:
Harmonic mean of Precision and recall:
`F1 = TP / (TP + (FN + FP)/2)`

### Models that solve a single task
.. _regression-reg:
##### Regression (Reg)
.. _default-dataset-reg1:
###### Default dataset (Reg1)
- [x] [INet](#inet)
- [x] [MobileNet](#mobilenet)
- [x] [VGG-16](#vgg-16)

####### INet
![image](../../wiki/uploads/0e5cfae73a1f4cc2a8c30742b22f0a4d/image.png)

```
###########################
	GIoU Loss:	 0.6161582 
	RMSE Loss:	 25.917557
```

![image](../../wiki/uploads/6896b06aca9e1a18ea0eb97035df4c7b/image.png)

####### MobileNet
![image](../../wiki/uploads/0a9943e8692fe8d7ba88096150047e35/image.png)

```
###########################
	GIoU Loss:	 0.61227506 
	RMSE Loss:	 25.372694
```
![image](../../wiki/uploads/13356e7e5e237e71e8834beecda54970/image.png)


####### VGG-16
![image](../../wiki/uploads/ec3cce9d8b3b0aab76a0ad856f1975ed/image.png)

```
###########################
	GIoU Loss:	 0.947507 
	RMSE Loss:	 24.218405
```
![image](../../wiki/uploads/0a0256f68878fb09f152b230dbf951e1/image.png)

Result:

|    Arch    | GIoU     |    RMSE   |
|:----------:|:--------:|:---------:|
|    INet    |  0.3838  |  25.9176  |
| MobileNet  | *0.3877* | *25.3727* |
|   VGG-16   |  0.0525  |  24.2184  |


.. _augmented-dataset-reg2:
###### Augmented dataset (Reg2)
- [x] [INet](#inet-1)
- [x] [MobileNet](#mobilenet-1)
- [x] [VGG-16](#vgg-16-1)

####### INet


  
![image](../../wiki/uploads/3bf030e1a6b8a908bb549f38fd0df80b/image.png)
```
###########################
	GIoU Loss:	 0.6330171 
	RMSE Loss:	 26.344194
```
![image](../../wiki/uploads/03052b00c9e8e3c394efa9b282065954/image.png)


###### MobileNet

![image](../../wiki/uploads/561fb982df99d4a0fba7eed21f0c7c34/image.png)
```
##########################
	GIoU Loss:	 0.43982178 
	RMSE Loss:	 17.006126
```
![image](../../wiki/uploads/6c89c40cbf4247728dfb27eadbe98b7b/image.png)


###### VGG-16

![image](../../wiki/uploads/d0e0a34d80f0943c8b77a12cee0b360f/image.png)
```
##########################
	GIoU Loss:	 1.399976 
	RMSE Loss:	 40.743893
```

![image](../../wiki/uploads/9709ec7e8d054ccdb366b0b44191705e/image.png)


Result:

|    Arch    | GIoU     |    RMSE   |
|:----------:|:--------:|:---------:|
|    INet    |  0.3670  |  26.3442  |
| MobileNet  | *0.5602* | *17.0061* |
|   VGG-16   | -0.4000  |  40.7438  |


### Classification (Clf)
.. _default-dataset-clf1:
##### Default dataset (Clf1)
- [x] [INet](#inet-2)
- [x] [MobileNet](#mobilenet-2)
- [x] [VGG-16](#vgg-16-2)

###### INet


  
![image](../../wiki/uploads/66e727d6d0ecbc37079ab6d5fa1e1083/image.png)
![image](../../wiki/uploads/b20ac42dc5513acdb43b7b7ae62063cb/image.png)

```
##########################
	Accuracy:	 0.6422222222222222 
	f1 score:	 0.6353292590535011
```
![image](../../wiki/uploads/5f16b9b17b2b0c63cc580d62485e8641/image.png)


###### MobileNet


  
![image](../../wiki/uploads/035490979d7229400b3c25a64e4f865b/image.png)
![image](../../wiki/uploads/64ac5e1b369f7d2ccf6df69e3031dc80/image.png)

```
##########################
	Accuracy:	 0.8511111111111112 
	f1 score:	 0.8462153544896847
```
![image](../../wiki/uploads/86b02c196ac86228ee60b36aac76572f/image.png)


###### VGG-16


  
![image](../../wiki/uploads/4beec2f007c6685cf48da0ceebd67d9c/image.png)
![image](../../wiki/uploads/e9258d94d5b7fee89a24c4583351ab2f/image.png)

```
##########################
	Accuracy:	 0.84 
	f1 score:	 0.8367397903530476
```
![image](../../wiki/uploads/f1c0cb86ea4a30d76b84b455f2014103/image.png)
Result:

|    Arch     |  Accuracy  |    f1    |
|:-----------:|:----------:|:--------:|
|     INet    |    0.6422  |  0.7793  |
|**MobileNet**| **0.8511** |**0.8462**|
|    VGG-16   |    0.84    |  0.8367  |


.. _augmented-dataset-clf2:
##### Augmented dataset (Clf2)
- [x] [INet](#inet-3)
- [x] [MobileNet](#mobilenet-3)
- [x] [VGG-16](#vgg-16-3)

###### INet


  
![image](../../wiki/uploads/d4917858e460c817eafa3f211dbd5a18/image.png)

![image](../../wiki/uploads/d054435fe709472f2abfd451dc3aaf3d/image.png)

```
##########################
	Accuracy:	 0.6422222222222222 
	f1 score:	 0.62984677235293
```
![image](../../wiki/uploads/44033735747e24bf623e14b99602503e/image.png)


###### MobileNet


  
![image](../../wiki/uploads/8ece3cb4eb757dd0bfc75ac7d299a00c/image.png)
![image](../../wiki/uploads/1ca908e26e871a14e8154400a1119c1c/image.png)

```
##########################
	Accuracy:	 0.8333333333333334 
	f1 score:	 0.8261255977635148
```
![image](../../wiki/uploads/8af1a4270d72758e85ddc1c4692f2a7b/image.png)


###### VGG-16


  
![image](../../wiki/uploads/9c80d332359881f9bbaec01d78c502b3/image.png)
![image](../../wiki/uploads/2b242a634de4637e930cf50c851990e9/image.png)
```
##########################
	Accuracy:	 0.8466666666666667 
	f1 score:	 0.8438866903702877
```
![image](../../wiki/uploads/a4e3e0f5233da6528f023c1012b2e086/image.png)


Result:
|    Arch     |  Accuracy  |    f1    |
|:-----------:|:----------:|:--------:|
|     INet    |   0.6422   |  0.6298  |
|**MobileNet**| **0.8333** |**0.8261**|
|    VGG-16   |   0.8467   |  0.8439  |

.. _uncropped-dataset-clf3:
### Uncropped dataset (Clf3)
- [x] [MobileNet](#mobilenet-4)

##### MobileNet

![image](../../wiki/uploads/f3d7ad410ec85c4bfac0c966f971bf78/image.png)
![image](../../wiki/uploads/c5dda8dcfe62a2db33634af85642bce6/image.png)
```
##########################
	Accuracy:	 0.7844444444444445 
	f1 score:	 0.779331433620758
```

![image](../../wiki/uploads/be222c8d28e92e760cf131aa1d7cb63f/image.png)

Result:

|     Arch      |  Accuracy  |     f1     |
|:-------------:|:----------:|:----------:|
| **MobileNet** | **0.7844** | **0.7793** |

### Models that solve two tasks at the same time

### 2-Head-Predictor (2Head)

##### Default dataset (2Head1)
- [x] [INet](#inet-4)
- [x] [MobileNet](#mobilenet-5)
- [x] [VGG-16](#vgg-16-4)

###### INet


  

![image](../../wiki/uploads/65432715e23355f56e66bdce49d610a6/image.png)
```
##########################
Regression: 
	GIoU Loss:	 0.64344555 
	RMSE Loss:	 24.211178 
Classification: 
	Accuracy:	 0.49333333333333335 
	f1 score:	 0.4762167417804296
```
![image](../../wiki/uploads/7cbce192c97e4c49e6a47a5bfb3165c6/image.png)
![image](../../wiki/uploads/59fe46a8384a1a8243be01bc7b463544/image.png)


###### MobileNet

  
![image](../../wiki/uploads/ea5c7e3963deacbd6e0fd185688de1bf/image.png)
```
##########################
Regression: 
	GIoU Loss:	 1.4751697 
	RMSE Loss:	 39.193527 
Classification: 
	Accuracy:	 0.7311111111111112 
	f1 score:	 0.7224772507688828
```
![image](../../wiki/uploads/7b986b137af03555a668b630cc4f6fbd/image.png)
![image](../../wiki/uploads/a54fa89f1d33235ba17d8f3af3dd1c71/image.png)

###### VGG-16


  
![image](../../wiki/uploads/c0eba9c1ba640d38581aca1568560c29/image.png)
```
##########################
Regression: 
	GIoU Loss:	 1.5730777 
	RMSE Loss:	 43.680496 
Classification: 
	Accuracy:	 0.17333333333333334 
	f1 score:	 0.0590909090909091
```
![image](../../wiki/uploads/ae242635517f0f1ac09c1f4d32e5576e/image.png)
![image](../../wiki/uploads/a012231f08e869cbf401fda07db7e25f/image.png)


Result:

Classification:

|     Arch      |  Accuracy  |     f1     |
|:-------------:|:----------:|:----------:|
|     INet      |   0.4933   |   0.4762   |
| **MobileNet** | **0.7311** | **0.7793** |
|    VGG-16     |   0.1733   |   0.0591   |

Localization:

|   Arch    |    GIoU    |    RMSE     |
|:---------:|:----------:|:-----------:|
| **INet**  | **0.3566** | **24.2112** |
| MobileNet |  -0.4751   |   39.1935   |
|  VGG-16   |  -0.5731   |   43.6805   |

##### Augmented dataset (2Head2)
- [x] [INet](#inet-5)
- [x] [MobileNet](#mobilenet-6)
- [x] [VGG-16](#vgg-16-5)

###### INet

![image](../../wiki/uploads/07b2c2bedbfe6f728fac3918d9b0a408/image.png)

```
##########################
Regression: 
	GIoU Loss:	 0.6958105 
	RMSE Loss:	 21.909544 
Classification: 
	Accuracy:	 0.48444444444444446 
	f1 score:	 0.4625029774807487
```
![image](../../wiki/uploads/f1361c38c1087356bbb3b7c0bd72e1ca/image.png)
![image](../../wiki/uploads/7720db08246ae82973715d7882643174/image.png)

###### MobileNet


  
![image](../../wiki/uploads/01dc1b53cfb009e78db70dffc9da015b/image.png)
```
##########################
Regression: 
	GIoU Loss:	 0.63962024 
	RMSE Loss:	 20.518587 
Classification: 
	Accuracy:	 0.7555555555555555 
	f1 score:	 0.7496034996537605
```
![image](../../wiki/uploads/0b4b8d06298c2afeaefd25d28a50c418/image.png)
![image](../../wiki/uploads/18321de5c6cac14e4148b2c226b219ec/image.png)


###### VGG-16


  
![image](../../wiki/uploads/7410591e84a83f5edaae0fdfec15904d/image.png)
```
##########################
Regression: 
	GIoU Loss:	 1.4623789 
	RMSE Loss:	 38.729637 
Classification: 
	Accuracy:	 0.78 
	f1 score:	 0.77794664790425
```
![image](../../wiki/uploads/6f9b4840ee990bde6c0f9cb7112ef759/image.png)
![image](../../wiki/uploads/7a2d15698b4a511b5115a2c23fb980d7/image.png)


Result:

Classification:

|     Arch      |  Accuracy  |     f1     |
|:-------------:|:----------:|:----------:|
|     INet      |   0.4844   |   0.4625   |
| **MobileNet** | **0.7555** | **0.7496** |
|    VGG-16     |   0.7800   |   0.7779   |

Localization:

|     Arch      |    GIoU    |     RMSE     |
|:-------------:|:----------:|:------------:|
|     INet      |   0.3042   |   24.2111    |
| **MobileNet** | **0.3604** | **20.5186**  |
|    VGG-16     |  -0.4624   |   38.7296    |

### YOLO/MobileNet-SSD (SSD)
- [ ] default dataset
- [ ] augmented dataset

Result:

### Comparisons
#### (best Reg) MobileNet + (best Clf) MobileNet 
  
```
Classification:
 #########################
	Accuracy:	 0.5288888888888889 
	f1 score:	 0.5097587081088926
```
![image](../../wiki/uploads/01f4f10deda9bbd214ffbedf284aa5f4/image.png)

![image](../../wiki/uploads/ff76205cdb0cc83d0974f31779581a4b/image.png)


Result:

- `2-S`: 2 Stage method  
- `2-S-1`: 2 Stage without retraining (BBReg: MobileNet, Clf: MobileNet)  
- `2-S-2`: 2 Stage with clf retrained (BBReg: MobileNet, Clf: INet)  
- `2-S-3`: 2 Stage with clf retrained  (BBReg: MobileNet, Clf: MobileNet)  
- `2-Head`: 2-Head MobileNet
- `Clf3`: Classifier trained on original images

Classification:

|  Arch  | Accuracy |    f1    |
|:------:|:--------:|:--------:|
|  Clf3  |  0.7844  |  0.7793  |
| 2-S-1  |  0.5289  |  0.5097  |
| 2-S-2  |  0.5289  |  0.5269  |
| 2-S-3  |  0.5666  |  0.5661  |
| 2-Head |  0.7555  |  0.7496  |

Localization:

|   Arch   |  GIoU  |    RMSE    |
|:--------:|:------:|:----------:|
|   Clf3   |   -    |     -      |
|  2-S-1   | 0.3877 |  25.3727   |
|  2-S-2   | 0.3877 |  25.3727   |
|  2-S-3   | 0.3877 |  25.3727   |
|  2-Head  | 0.3604 |  20.5186   |


