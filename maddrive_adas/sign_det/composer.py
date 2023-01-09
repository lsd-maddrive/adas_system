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

    def detect_and_classify_batch(
        self,
        imgs: List[np.ndarray],
        **kwargs
    ) -> List[Tuple[DetectedInstance, Tuple[str, float]]]:

        detections: List[DetectedInstance] = self._detector.detect_batch(imgs, **kwargs)
        classification_res_per_detected_instaces: List = \
            self._classifier.classify_batch(detections, **kwargs)

        return classification_res_per_detected_instaces

    def detect_and_classify(
        self,
        img: np.ndarray,
        **kwargs
    ) -> Tuple[DetectedInstance, List[Tuple[str, float]]]:
        return self.detect_and_classify_batch([img], **kwargs)[0]
