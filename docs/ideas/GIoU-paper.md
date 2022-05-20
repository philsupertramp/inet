# GIoU

- IoU commonly used for object detection benchmarks
- IoU as metric detached from common regression l1 and l2 distance losses
- IoU can be used as loss for axis aligned bbs
- problems generating metric for not overlapping predictions
- replaces l1 and l2 norms with IoU metric


## IoU
encodes widths, heights and locations of two bbs into region properties and calculates a normalized measure focused on their areas.
### Issues
can not be used as loss due to it's lack of value in cases without overlapping areas
### Object detection accuracy
Detects true positives and false positives in a set of predictions.
IoU is used with an accuracy measure
### Bounding box representation and losses


## general regression losses
### Issues
Calculation may suffer from encoding, e.g. $[x_c, y_c, w, h]$ with $x_c, y_c$ the center coordinates and $w, h$ width and height respectively.

## Generalized IoU
- follows same encoding definition as IoU
- maintains scale invariant property
- strong correlation with IoU in case of overlapping objects

## Algorithm
in `BoundingBox`

```python
def GIoU(self, other: 'BoundingBox') -> float:
  overlap_area = self.overlap(other).area
  iou = self.IoU(other)
  return iou - (overlap_area-self.A_U(other))/overlap_area
```
