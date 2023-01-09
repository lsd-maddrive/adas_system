from pathlib import Path

import pytest

from maddrive_adas.utils import fs
from maddrive_adas.utils import get_project_root
from maddrive_adas.sign_det.detector import YoloV5Detector
from maddrive_adas.sign_det.base import DetectedInstance


PROJECT_ROOT = get_project_root()
SIGN_DETECTOR_MODEL_ARCHIVE = str(
    PROJECT_ROOT / 'detector_archive'
)

detector = YoloV5Detector(
    model_archive_file_path=SIGN_DETECTOR_MODEL_ARCHIVE,
    conf_thres=0.5,
    iou_thres=0.5,
    device='cpu'
)


@pytest.fixture
def test_data_path():
    test_data_path = Path(__file__).parent / "test_data" / "test_detector_classifier"
    return test_data_path


@pytest.fixture
def detector_test_image1(test_data_path: Path):
    img_fpath = test_data_path / 'test_image.png'
    img = fs.imread_rgb(img_fpath)
    return img


@pytest.mark.detector
def test_detector_base_execution_img1(detector_test_image1):
    detection = detector.detect(detector_test_image1)
    assert isinstance(detection, DetectedInstance)


@pytest.mark.detector
def test_detector_base_execution_batch(
    detector_test_image1
):
    detections = detector.detect_batch(
        [detector_test_image1, detector_test_image1])
    assert len(detections) == 2


@pytest.mark.detector
def test_detector_base_execution_batch_empty():
    detections = detector.detect_batch([])
    assert not detections
