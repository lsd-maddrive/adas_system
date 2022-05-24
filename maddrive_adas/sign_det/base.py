import numpy as np
from typing import List, Tuple


class DetectedSign:
    """
    Class to control detected sign operations.

    We define one point of signs retectino contract for communication
    """

    def __init__(self, bbox: List[float]) -> None:
        self._bbox = np.array(bbox, dtype=np.float32)

    def as_dict(self) -> dict:
        return {"bbox": self._bbox.tolist()}


class BaseSignsDetector:
    """Base Detector."""

    def __init__(self) -> None:
        pass

    def detect(self, img: np.array) -> dict:
        predictions = self.detect_batch([img])
        return predictions[0]

    def detect_batch(self, imgs: List[np.array]) -> List[dict]:
        raise NotImplementedError()


class BaseSignsClassifier:
    """Base Calssifier."""
# TODO: fix constructor
    def __init__(self) -> None:
        pass

    def classify(self, imgs: List[np.array]) -> List[Tuple[str, float]]:
        raise NotImplementedError()
