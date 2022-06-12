from pathlib import Path
import torch

import numpy as np

from src.utils.fs import imread_rgb

from base import AbstractSignClassifier, AbstatractSignDetector, AbstractComposer
from detector import YoloV5Detector
from classifier import EncoderBasedClassifier


class YoloSignsDetector():
    """Signs  detector base on YOLO"""

    def __init__(
        self,
        path_to_classifier_config_data: str,
        path_to_detector_config_data: str,
        not_encoder_based_classifier: bool = True,
        device: torch.device = None,
    ):
        if not_encoder_based_classifier:
            self._classifier: AbstractSignClassifier = EncoderBasedClassifier(
                config_path=path_to_classifier_config_data
            )
        else:
            assert False, 'Not implemented'

        self._detector: AbstatractSignDetector = YoloV5Detector(
            config_path=path_to_detector_config_data
        )

    def detect_and_classify_batch(self, imgs: list[np.array]) -> list[dict]:
        DEBUG = True
        if DEBUG:
            import cv2

        detection_res = self._detector.detect_batch(imgs)
        classification_res_list = []
        for detection_per_single_img in detection_res:
            classification_res = self._classifier.classify(detection_per_single_img)
            classification_res_list.append(classification_res)

            if DEBUG:
                for img, sign in zip(detection_per_single_img, classification_res):
                    img_ = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
                    cv2.imshow(sign[0], img_)
                cv2.waitKey()
        return classification_res

    def detect_and_classify(self, img: np.array) -> dict:
        return self.detect_and_classify_batch([img])


def test():
    PROJECT_ROOT = Path('.')
    DATA_DIR = PROJECT_ROOT / 'tests' / 'test_data'
    DETECTOR_ARCHIVE = PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'detector_config_img_size'
    CLASSIFIER_ARCHIVE = PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'encoder_cl_config'

    img1 = imread_rgb(DATA_DIR / 'custom_test.png')
    img2 = imread_rgb(DATA_DIR / 'test_image.png')

    model: AbstractComposer = YoloSignsDetector(
        path_to_classifier_config_data=CLASSIFIER_ARCHIVE,
        path_to_detector_config_data=DETECTOR_ARCHIVE
    )

    res = model.detect_and_classify_batch(
        [img1, img2]
    )
    return res


if __name__ == '__main__':
    test()
