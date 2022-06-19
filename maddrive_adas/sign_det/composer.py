import numpy as np

from .base import (
    AbstractSignClassifier, AbstractSignDetector,
    AbstractComposer, DetectedInstance
)


class BasicSignsDetectorAndClassifier(AbstractComposer):
    """Signs  detector base on YOLO"""

    def __init__(
        self,
        detector: AbstractSignDetector,
        classifier: AbstractSignClassifier,
    ):
        self._detector: AbstractSignDetector = detector
        self._classifier: AbstractSignClassifier = classifier

    def detect_and_classify_batch(self, imgs: list[np.array]) -> list[dict]:

        detections: list[DetectedInstance] = self._detector.detect_batch(imgs)
        classification_res = self._classifier.classify_batch(detections)

        return classification_res

    def detect_and_classify(self, img: np.array) -> dict:
        return self.detect_and_classify_batch([img])
