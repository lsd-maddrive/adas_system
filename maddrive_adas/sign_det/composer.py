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
        classification_res = self._classifier.classify_batch(detections)

        # TODO: remove this shiet
        # in case we have 2 DetectedInstance in 3 images per each
        # classifier will return list of 6 results.
        # So rearrange classificatio_res per DetectedInstance
        res_per_detected_instance: List[DetectedInstance, List[Tuple[str, float]]] = []
        accum: int = 0
        for d in detections:
            roi_count: int = d.get_roi_count()
            res_per_detected_instance.append(
                (
                    d,
                    [x for x in classification_res[accum: accum + roi_count]]
                )
            )
            accum += roi_count

        return res_per_detected_instance

    def detect_and_classify(self, img: np.array) -> Tuple[DetectedInstance, Tuple[str, float]]:
        return self.detect_and_classify_batch([img])[0]
