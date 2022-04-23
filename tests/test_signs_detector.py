import os
import pytest
import numpy as np

from maddrive_adas.sign_det.yolo_detector import YoloSignsDetector
from maddrive_adas.src.utils import fs


@pytest.fixture(scope="module")
def detector_test_image(test_data_dpath) -> np.array:
    img_fpath = os.path.join(test_data_dpath, "test_image.png")
    img = fs.imread_rgb(img_fpath)
    return img


def test_detector_bsae_execution(detector_test_image):
    # .copy() - to avoid overwriting over source image
    src_img = detector_test_image.copy()

    detector = YoloSignsDetector()

    detections = detector.detect(src_img)

    assert len(detections) == 2
    # TODO - other checks like coordinates (position), width/height (size)
