import os
import pytest
import numpy as np

from maddrive_adas.sign_det.yolo_detector import YoloSignsDetector
from maddrive_adas.sign_det.src.utils import fs


@pytest.fixture(scope="module")
def detector_test_image(test_data_dpath) -> np.array:
    img_fpath = os.path.join(test_data_dpath, "test_image.png")
    img = fs.imread_rgb(img_fpath)
    return img


def test_detector_bsae_execution(detector_test_image):
    # .copy() - to avoid overwriting over source image
    src_img = detector_test_image.copy()

    detector = YoloSignsDetector()
    detector.initialize(
        path_to_yolo_cfg='yolov5L_custom_anchors.yaml',
        path_to_yolo_weights='YoloV5L_weights.pt',
        path_to_classifier_weights='classifier_wights.pt',
        not_encoder_based_classifier=True,
        path_to_centroid_location='centroid_location.txt',
        path_to_model_config='encoder_config.json'
    )

    detections = detector.detect(src_img)

    assert len(detections) == 2
    # TODO - other checks like coordinates (position), width/height (size)
