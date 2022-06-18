import os
from pathlib import Path

import pytest
import numpy as np

from maddrive_adas.sign_det.composer import BasicSignsDetectorAndClassifier
from maddrive_adas.sign_det.base import AbstractSignDetector
from maddrive_adas.utils import fs


@pytest.fixture(scope="module")
def detector_test_image(test_data_dpath) -> np.array:
    img_fpath = os.path.join(test_data_dpath, "test_image.png")
    img = fs.imread_rgb(img_fpath)
    return img


def test_detector_base_execution(detector_test_image):
    # .copy() - to avoid overwriting over source image
    src_img = detector_test_image.copy()
    PROJECT_ROOT = Path(__file__).parents[2]
    MODEL_ARCHIVE = PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'detector_config_img_size'

    detector: AbstractSignDetector = BasicSignsDetectorAndClassifier(
        config_path=str(MODEL_ARCHIVE)
    )

    detections = detector.detect(src_img)

    assert len(detections) == 2
    # TODO - other checks like coordinates (position), width/height (size)


if __name__ == '__main__':
    test_detector_base_execution
