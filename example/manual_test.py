from pathlib import Path

from maddrive_adas.utils.fs import imread_rgb
from maddrive_adas.sign_det.base import (
    AbstractSignDetector, AbstractSignClassifier,
    AbstractComposer, DetectedInstance
)
from maddrive_adas.sign_det.detector import YoloV5Detector
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier
from maddrive_adas.sign_det.composer import BasicSignsDetectorAndClassifier

PROJECT_ROOT = Path('.')
DETECTOR_ARCHIVE = PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'detector_config_img_size'
CLASSIFIER_ARCHIVE = PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'encoder_cl_config'

c: AbstractSignClassifier = EncoderBasedClassifier(
    config_path=str(CLASSIFIER_ARCHIVE)
)

d: AbstractSignDetector = YoloV5Detector(
    config_path=str(DETECTOR_ARCHIVE)
)


def test_detector():
    DATA_DIR = PROJECT_ROOT / 'tests' / 'test_data'

    img1 = imread_rgb(DATA_DIR / 'custom_test.png')
    img2 = imread_rgb(DATA_DIR / 'test_image.png')

    sign = d.detect_batch([img1, img2])

    return sign


def test_classifier():
    DATA_DIR = PROJECT_ROOT / 'SignDetectorAndClassifier' / 'data'

    img1 = imread_rgb(DATA_DIR / 'additional_sign' / '2.4_1.png')
    img2 = imread_rgb(DATA_DIR / 'additional_sign' / '1.31_1.png')
    img3 = imread_rgb(DATA_DIR / 'additional_sign' / '3.24.100_3.png')

    classify_batch_arg: list[DetectedInstance] = [
        DetectedInstance(img1),
        DetectedInstance(img2),
        DetectedInstance(img3),
    ]
    for di in classify_batch_arg:
        di.add_rel_roi([0., 0, 1., 1.], 1.)
        di.show_img()

    sign = c.classify_batch(classify_batch_arg)

    return sign


def test_composer():
    DATA_DIR = PROJECT_ROOT / 'tests' / 'test_data'

    img1 = imread_rgb(DATA_DIR / 'custom_test.png')
    img2 = imread_rgb(DATA_DIR / 'test_image.png')

    model: AbstractComposer = BasicSignsDetectorAndClassifier(
        classifier=c,
        detector=d
    )

    res = model.detect_and_classify_batch(
        [img1, img2]
    )
    return res


if __name__ == '__main__':
    a = test_detector()
    b = test_classifier()
    c = test_composer()
    assert False, 'Check opened windows and fcns return values'
