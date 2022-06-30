from typing import List

from pathlib import Path

from maddrive_adas.utils.fs import imread_rgb
from maddrive_adas.sign_det.base import (
    AbstractSignClassifier, DetectedInstance
)
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier

PROJECT_ROOT = Path('.')
CLASSIFIER_ARCHIVE = PROJECT_ROOT / 'encoder_archive'

c: AbstractSignClassifier = EncoderBasedClassifier(
    config_path=str(CLASSIFIER_ARCHIVE)
)


def test_classifier():
    DATA_DIR = PROJECT_ROOT / 'SignDetectorAndClassifier' / 'data'

    # img1 = imread_rgb(DATA_DIR / 'additional_sign' / '2.4_1.png')
    # img2 = imread_rgb(DATA_DIR / 'additional_sign' / '1.31_1.png')
    img1 = imread_rgb(DATA_DIR / 'additional_sign' / '3.24.100_1.png')
    img2 = imread_rgb(DATA_DIR / 'additional_sign' / '3.24.100_2.png')
    img3 = imread_rgb(DATA_DIR / 'additional_sign' / '3.24.100_3.png')
    classify_batch_arg: List[DetectedInstance] = [
        DetectedInstance(img1),
        DetectedInstance(img2),
        DetectedInstance(img3),
    ]
    for di in classify_batch_arg:
        di.add_rel_roi([0., 0, 1., 1.], 1.)

    sign_multi = c.classify_batch(classify_batch_arg)
    sign_img = c.classify_batch([img1, img2])

    return sign_multi, sign_img


if __name__ == '__main__':
    b = test_classifier()
    assert False, 'Check opened windows or/and fcns return values'
