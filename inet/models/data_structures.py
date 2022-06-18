from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from matplotlib.patches import Rectangle
from tensorflow import keras

from inet.models.architectures.base_model import TaskModel


class BoundingBox:
    """
    BoundingBox representation to parse, draw and evaluate
    """
    def __init__(self, x: float, y: float, w: float, h: float):
        """
        :param x: x-min coordinate
        :param y: y-min coordinate
        :param w: width
        :param h: height
        """
        ## BBox parameter as vector
        self.values = [x, y, w, h]
        ## Minimal X-value
        self.x_min = self.values[0]
        ## Minimal Y-value
        self.y_min = self.values[1]
        ## BBox width
        self.w = self.values[2]
        ## BBox height
        self.h = self.values[3]

    @property
    def half_w(self) -> float:
        """
        :return: half of the bbs width
        """
        return self.w / 2.

    @property
    def half_h(self) -> float:
        """
        :return: half of the bbs height
        """
        return self.h / 2.

    @property
    def x_max(self) -> float:
        """
        :return: x-max coordinate
        """
        return self.x_min + self.w

    @property
    def y_max(self) -> float:
        """
        :return: y-max coordinate
        """
        return self.y_min + self.h

    @property
    def area(self) -> float:
        """

        :return:
        """
        return self.w * self.h

    def A_I(self, other: 'BoundingBox') -> float:
        """
        Area of intersection with another BBox `other`
        :param other: a different BBox
        :return: the area of intersection
        """
        width = abs(min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        height = abs(min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        return width * height

    def A_U(self, other: 'BoundingBox') -> float:
        """
        Area of union with other BBox `other`
        :param other: a different BBox
        :return: the area of union
        """
        return self.area + other.area - self.A_I(other)

    def IoU(self, other: 'BoundingBox') -> float:
        """
        Computes the intersection over union (IoU) with a different BBox `other`
        :param other: a different BBox
        :return: intersection over union value
        """
        return self.A_I(other)/self.A_U(other)

    def overlap(self, bb2: 'BoundingBox') -> Optional['BoundingBox']:
        """
        Generates an overlapping BBox/convex hull around `self` and `bb2`
        :param bb2: a different BBox
        :return: when overlapping, a new BBox containing both BBoxes
        """
        out = BoundingBox(
            max(self.x_min, bb2.x_min),
            max(self.y_min, bb2.y_min),
            min(self.x_max, bb2.x_max),
            min(self.y_max, bb2.y_max)
        )
        if out.x_min > out.x_max or out.y_min > out.y_max:
            return None
        return out

    def GIoU(self, other: 'BoundingBox') -> float:
        """
        Generalized intersection over union (GIoU) based on [the paper](https://giou.stanford.edu/GIoU.pdf)
        :param other: other BB to compute GIoU with
        :return: GIoU for `self` and `other`
        """
        convex_hull_area = self.overlap(other).area
        iou = self.IoU(other)
        if convex_hull_area == 0:
            return iou
        return iou - (convex_hull_area-self.A_U(other))/convex_hull_area

    def draw(self, gc, color: str = 'red') -> None:
        """
        Method to render BBox into a graphic-context `gc`
        :param gc: graphics-context, e.g. `matplotlib.pyplot.gca()`
        :param color: the color of the BBox
        :return:
        """
        gc.add_patch(
            Rectangle(
                (self.x_min, self.y_min),
                self.w,
                self.h,
                linewidth=1.5,
                edgecolor=color,
                facecolor=color,
                fill=False
            )
        )

    def __str__(self) -> str:
        """string representation"""
        return f'BoundingBox [{self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}]'

    def __eq__(self, other: 'BoundingBox') -> bool:
        """Equals operator to compare two BBs"""
        return np.linalg.norm(np.array(self.values) - np.array(other.values)) <= np.finfo(np.float32).eps


@dataclass
class ModelArchitecture:
    """
    Helper dataclass to simplify creation of model architecture, mostly used in development notebooks.
    """
    ## The backbone to use
    backbone: keras.models.Sequential
    ## Name of the architecture
    name: str
    ## Callback to create a model out of the architecture
    create_model: Optional[
        Callable[[keras.models.Sequential, Optional[str], Optional[int], Optional[float]], TaskModel]
    ] = None
