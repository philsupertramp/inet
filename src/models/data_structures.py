from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from matplotlib.patches import Rectangle
from tensorflow import keras

from src.models.architectures.base_model import TaskModel


class BoundingBox:
    def __init__(self, x: float, y: float, w: float, h: float):
        self.values = [x, y, w, h]
        self.x_min = self.values[0]
        self.y_min = self.values[1]
        self.w = self.values[2]
        self.h = self.values[3]

    @property
    def half_w(self):
        return self.w / 2.

    @property
    def half_h(self):
        return self.h / 2.

    @property
    def x_max(self):
        return self.x_min + self.w

    @property
    def y_max(self):
        return self.y_min + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    def A_I(self, other: 'BoundingBox') -> float:
        width = abs(min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        height = abs(min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        return width * height

    def A_U(self, other: 'BoundingBox') -> float:
        return self.area + other.area - self.A_I(other)

    def IoU(self, other: 'BoundingBox') -> float:
        return self.A_I(other)/self.A_U(other)

    def overlap(self, bb2: 'BoundingBox') -> Optional['BoundingBox']:
        out = BoundingBox(
            max(self.x_min, bb2.x_min),
            max(self.y_min, bb2.y_min),
            min(self.x_max, bb2.x_max),
            min(self.y_max, bb2.y_max)
        )
        if out.x_min > out.x_max or out.y_min > out.y_max:
            return None
        return out

    def hull(self, other: 'BoundingBox') -> 'BoundingBox':
        return BoundingBox(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max),
        )

    def GIoU(self, other: 'BoundingBox') -> float:
        convex_hull_area = self.hull(other).area
        iou = self.IoU(other)
        if convex_hull_area == 0:
            return iou
        return iou - (convex_hull_area-self.A_U(other))/convex_hull_area

    def draw(self, gc, color: str = 'red') -> None:
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
        return f'BoundingBox [{self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}]'

    def __eq__(self, other: 'BoundingBox') -> bool:
        return np.linalg.norm(np.array(self.values) - np.array(other.values)) <= np.finfo(np.float32).eps


@dataclass
class ModelArchitecture:
    backbone: keras.models.Sequential
    name: str
    create_model: Optional[
        Callable[[keras.models.Sequential, Optional[str], Optional[int], Optional[float]], TaskModel]
    ] = None
