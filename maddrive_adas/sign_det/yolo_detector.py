from typing import List

import numpy as np

import sys
import os

from classifier import EncoderBasedClassifier, NonEncoderBasedClassifier
from detector import YoloV5Detector

from base import BaseSignsClassifier, DetectedSign, BaseSignsDetector
from src.utils.logger import logger

# TODO
# append maddrive_adas
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# append src as root
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))


class YoloSignsDetector(BaseSignsDetector):
    """Signs  detector base on YOLO"""

    def __init__(self) -> None:

        import torch
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
        class_count: int,
        not_encoder_based_classifier: bool = True
    ) -> bool:

        try:
            if not_encoder_based_classifier:
                self._classifier = NonEncoderBasedClassifier
            else:
                self._classifier = EncoderBasedClassifier

            self._classifier = self._classifier(
                path_to_classifier_weights,
                class_count,
                self._device
            )

            # TODO
            # FIX HARDCODED img_size, use_half
            self._detector = YoloV5Detector(
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
        # Sample code
        # TODO - replace for real one
        predictions = [DetectedSign(bbox=[0, 10, 0, 10])]

        return [ds.as_dict() for ds in predictions]


def test():
    model = YoloSignsDetector()
    path_prefix = os.path.dirname(__file__) + '\\'
    model.initialize(
        path_to_yolo_cfg=path_prefix + 'yolov5l_model_config.yaml',
        path_to_yolo_weights=path_prefix + 'YoloV5L_weights.pt',
        # path_to_classifier_weights=path_prefix + 'resnet_CLASSIFIER_ON_STOCK',
        # class_count=57,
        # not_encoder_based_classifier=True
        path_to_classifier_weights=path_prefix + 'encoder_model_only',
        class_count=512,
        not_encoder_based_classifier=False
    )


if __name__ == '__main__':
    test()
