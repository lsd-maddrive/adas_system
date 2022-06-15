from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))  # TODO: fix dynamic appending

import numpy as np

from src.utils.fs import imread_rgb
from base import (
    AbstractSignClassifier, AbstatractSignDetector,
    AbstractComposer, DetectedInstance
)
from classifier import EncoderBasedClassifier
from detector import YoloV5Detector


class YoloSignsDetectorAndClassifier(AbstractComposer):
    """Signs  detector base on YOLO"""

    def __init__(
        self,
        detector: AbstatractSignDetector,
        classifier: AbstractSignClassifier,
    ):
        self._detector: AbstatractSignDetector = detector
        self._classifier: AbstractSignClassifier = classifier

    def detect_and_classify_batch(self, imgs: list[np.array]) -> list[dict]:

        detections: list[DetectedInstance] = self._detector.detect_batch(imgs)
        classification_res = self._classifier.classify_batch(detections)

        return classification_res

    def detect_and_classify(self, img: np.array) -> dict:
        return self.detect_and_classify_batch([img])


def test():
    PROJECT_ROOT = Path(__file__).parents[2]
    DATA_DIR = PROJECT_ROOT / 'tests' / 'test_data'
    DETECTOR_ARCHIVE = PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'detector_config_img_size'
    CLASSIFIER_ARCHIVE = PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'encoder_cl_config'

    img1 = imread_rgb(DATA_DIR / 'custom_test.png')
    img2 = imread_rgb(DATA_DIR / 'test_image.png')

    c: AbstractSignClassifier = EncoderBasedClassifier(
        config_path=str(CLASSIFIER_ARCHIVE)
    )

    d: AbstatractSignDetector = YoloV5Detector(
        config_path=str(DETECTOR_ARCHIVE)
    )

    model: AbstractComposer = YoloSignsDetectorAndClassifier(
        classifier=c,
        detector=d
    )

    res = model.detect_and_classify_batch(
        [img1, img2]
    )
    return res


if __name__ == '__main__':
    test()
