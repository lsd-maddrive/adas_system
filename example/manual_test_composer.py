from pathlib import Path

from maddrive_adas.utils.fs import imread_rgb
from maddrive_adas.sign_det.composer import BasicSignsDetectorAndClassifier
from maddrive_adas.sign_det.base import (
    AbstractComposer, AbstractSignClassifier,
    AbstractSignDetector
)
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier
from maddrive_adas.sign_det.detector import YoloV5Detector

PROJECT_ROOT = Path('.')
# FIX THIS PATH BY YOURS
DETECTOR_ARCHIVE = PROJECT_ROOT / 'detector_archive'
CLASSIFIER_ARCHIVE = PROJECT_ROOT / 'encoder_archive'

c: AbstractSignClassifier = EncoderBasedClassifier(
    config_path=str(CLASSIFIER_ARCHIVE)
)

d: AbstractSignDetector = YoloV5Detector(
    config_path=str(DETECTOR_ARCHIVE)
)

composer: AbstractComposer = BasicSignsDetectorAndClassifier(
    classifier=c,
    detector=d
)


def test_composer():
    DATA_DIR = PROJECT_ROOT / 'tests' / 'test_data'
    img1 = imread_rgb(DATA_DIR / 'custom_test.png')
    res = composer.detect_and_classify(img1)

    return res


if __name__ == '__main__':
    b = test_composer()
    assert False, 'Check opened windows or/and fcns return values'
