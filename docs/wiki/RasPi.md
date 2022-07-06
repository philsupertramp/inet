## Inference tests
### Original:
```text
Model evaluation for: "independent":
Classification:
 ===================================
	Accuracy:	0.916
	f1 score:	0.9167668857681328
Localization:
 ===================================
	GIoU:	0.43616188
	RMSE:	17.347866
Testing inference for 5 samples for: "independent":
AVG inference time: 2.4359699999999997s.

Model evaluation for: "two-stage":
Classification:
 ===================================
	Accuracy:	0.544
	f1 score:	0.5522557272067077
Localization:
 ===================================
	GIoU:	0.43616188
	RMSE:	17.347866
Testing inference for 5 samples for: "two-stage":
AVG inference time: 3.467743s.

Model evaluation for: "single-stage":
Classification:
 ===================================
	Accuracy:	0.92
	f1 score:	0.9196824463343246
Localization:
 ===================================
	GIoU:	0.62672853
	RMSE:	19.672493
Testing inference for 5 samples for: "single-stage":
AVG inference time: 1.2404s.
```

### TFLite:
```text
Model evaluation for: "independent":
Classification:
 ===================================
	Accuracy:	0.92
	f1 score:	0.9209747544826291
Localization:
 ===================================
	GIoU:	0.44598112
	RMSE:	18.026241
Testing inference for 5 samples for: "independent":
AVG inference time: 1.324131s.


Model evaluation for: "two-stage":
Classification:
 ===================================
	Accuracy:	0.52
	f1 score:	0.5284798070186199
Localization:
 ===================================
	GIoU:	0.44598112
	RMSE:	18.026241
Testing inference for 5 samples for: "two-stage":
AVG inference time: 2.34496s.


Model evaluation for: "single-stage":
[====================] 100%
Classification:
 ===================================
	Accuracy:	0.608
	f1 score:	0.5803820077663839
Localization:
 ===================================
	GIoU:	0.81673026
	RMSE:	21.786606
Testing inference for 5 samples for: "single-stage":
AVG inference time: 0.731425s.
```
