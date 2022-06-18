from enum import Enum
from typing import Any, Dict, Iterable, NewType, Tuple, Union

# Unfortunately it becomes very cryptic once tf.Tensor and np.array are involved.
# Therefore, single elements will receive the type Any
ImageType = NewType('ImageType', Any)
ClassLabelType = NewType('ClassLabelType', str)
BoundingBoxLabelType = NewType('BoundingBoxLabelType', Tuple[float, float, float, float])
Features = Iterable[Any]
LabelType = Union[ClassLabelType, BoundingBoxLabelType, Tuple[ClassLabelType, BoundingBoxLabelType]]
Labels = Iterable[LabelType]

BBoxLabelType = Dict[str, float]
ClassLabelType = str
LabeledFileType = Dict[str, Union[BBoxLabelType, ClassLabelType]]
LabeledFileDictType = Dict[str, Dict[str, Union[BBoxLabelType, ClassLabelType]]]


class LabelType(Enum):
    ## No label expected
    NONE = 0
    ## Single label (BBox or class label)
    SINGLE = 1
    ## Multiple labels (BBox and class label)
    MULTI = 2


class ModelType(Enum):
    ## Regression model type
    REGRESSION = 0
    ## Classification model type
    CLASSIFICATION = 1
    ## Mixed two-in-one model type
    TWO_IN_ONE = 2
