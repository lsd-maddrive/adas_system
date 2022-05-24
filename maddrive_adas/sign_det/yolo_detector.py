from typing import List

import sys
import os
import torch

import numpy as np

# append src as root
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from classifier import NonEncoderBasedClassifier
from detector import YoloV5Detector

from base import BaseSignsClassifier, BaseSignsDetector
from src.utils.logger import logger
from src.utils.fs import imread_rgb


class YoloSignsDetector(BaseSignsDetector):
    """Signs  detector base on YOLO"""

    def __init__(self) -> None:

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self._is_initialized: bool = False
        self._detector: BaseSignsDetector = None
        self._classifier: BaseSignsClassifier = None

    def initialize(
        self,
        path_to_yolo_cfg: str,
        path_to_yolo_weights: str,
        path_to_classifier_weights: str,
        not_encoder_based_classifier: bool = True,
        device: torch.device = None,
        **kwargs
    ) -> bool:

        if device:
            logger.info(
                f'{__name__} Initialize. Overriding device -> {device.type}'
            )
            self._device = device

        try:
            if not_encoder_based_classifier:
                self._classifier = NonEncoderBasedClassifier(
                    path_to_classifier_weights,
                    path_to_model_config=kwargs['path_to_model_config'],
                    path_to_centroid_location=kwargs['path_to_centroid_location'],
                    device=self._device
                )
            else:
                # self._classifier = EncoderBasedClassifier
                pass
            # TODO
            # FIX HARDCODED img_size, use_half
            self._detector: YoloV5Detector = YoloV5Detector(
                path_to_cfg=path_to_yolo_cfg,
                path_to_weights=path_to_yolo_weights,
                device=self._device,
                img_size=(640, 640),
                use_half=True
            )

            return True

        except (FileNotFoundError, KeyError, RuntimeError) as exc_obj:
            import traceback
            exception_verbose = ''.join(
                traceback.format_exception(
                    None,
                    exc_obj,
                    exc_obj.__traceback__
                )
            )
            logger.error(
                f'{__name__}: Error occur while initialize {exception_verbose}'
            )

        return False

    def detect_batch(self, imgs: List[np.array]) -> List[dict]:
        DEBUG = False
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


def test():
    model = YoloSignsDetector()
    path_prefix = os.path.dirname(__file__) + '/'
    model.initialize(
        path_to_yolo_cfg=path_prefix + 'yolov5L_custom_anchors.yaml',
        path_to_yolo_weights="D:/d_tsw/main_diplom/SignDetectorAndClassifier/data/YoloV5L.pt",
        path_to_classifier_weights='EXCLUDE_ADDI_SIGNSencoder_loss_1e'
        '-05_acc_0.99566epoch_99.encoder',
        not_encoder_based_classifier=True,
        path_to_centroid_location='centroid_location.txt',
        path_to_model_config='encoder_config.json'
    )

    img = imread_rgb('../../tests/test_data/custom_test.png')
    img1 = imread_rgb('../../tests/test_data/test_image.png')
    res = model.detect_batch(
        [img, img1]
    )
    return res


if __name__ == '__main__':
    test()
