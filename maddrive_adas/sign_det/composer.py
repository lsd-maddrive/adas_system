from typing import List, Tuple

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

    def detect_and_classify_batch(self, imgs: List[np.array]) -> \
            List[Tuple[DetectedInstance, Tuple[str, float]]]:

        detections: List[DetectedInstance] = self._detector.detect_batch(imgs)
        classification_res_per_detected_instaces: List = \
            self._classifier.classify_batch(detections)

        return classification_res_per_detected_instaces

    def detect_and_classify(self, img: np.array) -> Tuple[DetectedInstance, Tuple[str, float]]:
        return self.detect_and_classify_batch([img])[0]
