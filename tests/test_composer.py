from pathlib import Path

import pytest

from maddrive_adas.utils import fs
from maddrive_adas.utils import get_project_root
from maddrive_adas.sign_det.detector import YoloV5Detector
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier
from maddrive_adas.sign_det.composer import BasicSignsDetectorAndClassifier
from maddrive_adas.sign_det.base import DetectedInstance

DEVICE = 'cpu'
PROJECT_ROOT = get_project_root()
SIGN_DETECTOR_MODEL_ARCHIVE: str = str(
    PROJECT_ROOT / 'detector_archive'
)
SIGN_CLASSIFIER_MODEL_ARCHIVE: str = str(
    PROJECT_ROOT / 'classifier_archive'
)

classifier = EncoderBasedClassifier(
    config_path=SIGN_CLASSIFIER_MODEL_ARCHIVE
)

detector = YoloV5Detector(
    model_archive_file_path=SIGN_DETECTOR_MODEL_ARCHIVE,
    conf_thres=0.5,
    iou_thres=0.5,
    device=DEVICE
)

composer = BasicSignsDetectorAndClassifier(
    classifier=classifier,
    detector=detector
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


@pytest.mark.composer
def test_composer_single_img(detector_test_image1):
    result = composer.detect_and_classify(detector_test_image1)
    assert isinstance(result, tuple)
    assert isinstance(result[0], DetectedInstance)  # detector related
    assert isinstance(result[1], list)              # classifier info
    assert isinstance(result[1][0], tuple)


@pytest.mark.composer
def test_composer_batch(detector_test_image1):
    result = composer.detect_and_classify_batch(
        [detector_test_image1, detector_test_image1]
    )
    assert isinstance(result, list)
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], DetectedInstance)


@pytest.mark.composer
def test_composer_empty_batch():
    result = composer.detect_and_classify_batch(
        []
    )
    assert not result
