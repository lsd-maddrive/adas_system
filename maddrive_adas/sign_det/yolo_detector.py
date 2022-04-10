import numpy as np
from typing import List

from .base import DetectedSign, BaseSignsDetector


class YoloSignsDetector(BaseSignsDetector):
    """Signs  detector base on YOLO"""

    def __init__(self) -> None:
        pass

    def detect_batch(self, imgs: List[np.array]) -> List[dict]:
        # Sample code
        # TODO - replace for real one
        predictions = [DetectedSign(bbox=[0, 10, 0, 10])]

        return [ds.as_dict() for ds in predictions]
