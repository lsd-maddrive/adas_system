from pathlib import Path

from maddrive_adas.sign_det.detector import YoloV5Detector
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier
from maddrive_adas.sign_det.composer import BasicSignsDetectorAndClassifier
from maddrive_adas.sign_det.base import (
    AbstractSignDetector, AbstractSignClassifier,
    AbstractComposer
)

PROJECT_ROOT: Path = Path('.')
SIGN_DETECTOR_MODEL_ARCHIVE: str = str(
    PROJECT_ROOT / 'detector_archive'
)
SIGN_CLASSIFIER_MODEL_ARCHIVE: str = str(
    PROJECT_ROOT / 'encoder_archive'
)

detector: AbstractSignDetector = YoloV5Detector(
    config_path=SIGN_DETECTOR_MODEL_ARCHIVE
)

classifier: AbstractSignClassifier = EncoderBasedClassifier(
    config_path=SIGN_CLASSIFIER_MODEL_ARCHIVE
)

composer: AbstractComposer = BasicSignsDetectorAndClassifier(
    classifier=classifier,
    detector=detector
)

# Detector


def test_detector_base_execution_img1(detector_test_image1):
    detections = detector.detect(
        detector_test_image1,
        d_conf_thres=0.11,
        d_iou_thres=0.12)
    assert len(detections.confs) == 2


def test_detector_base_execution_img2(detector_test_image2):
    detections = detector.detect(detector_test_image2)
    assert len(detections.confs) == 2


def test_detector_base_execution_batch(
    detector_test_image1,
    detector_test_image2
):
    detections = detector.detect_batch(
        [detector_test_image1, detector_test_image2])
    assert len(detections) == 2


def test_detector_base_execution_batch_empty():
    detections = detector.detect_batch([])
    assert len(detections) == 0

# Classifier


def test_classifier_base_2_1_1_img(classifier_test_img_2_1_2):
    classification_res = classifier.classify(classifier_test_img_2_1_2)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


def test_classifier_base_2_1_2_di(classifier_test_di_2_1_2):
    classification_res = classifier.classify(classifier_test_di_2_1_2)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


def test_classifier_base_5_19_img(classifier_test_img_5_19):
    classification_res = classifier.classify(classifier_test_img_5_19)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


def test_classifier_base_5_19_di(classifier_test_di_5_19):
    classification_res = classifier.classify(classifier_test_di_5_19)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


def test_classifier_base_7_4_img(classifier_test_img_7_4):
    classification_res = classifier.classify(classifier_test_img_7_4)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


def test_classifier_base_7_4_di(classifier_test_di_7_4):
    classification_res = classifier.classify(classifier_test_di_7_4)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


def test_classifier_batch_test(
    classifier_test_di_2_1_2,
    classifier_test_img_7_4,
    classifier_test_di_7_4
):
    classification_res = classifier.classify_batch(
        [
            classifier_test_di_2_1_2,
            classifier_test_img_7_4,
            classifier_test_di_7_4
        ]
    )
    assert len(classification_res) == 3


def test_classifier_batch_empty():
    classification_res = classifier.classify_batch([])
    assert len(classification_res) == 0


def test_composer_img1(detector_test_image1):
    result = composer.detect_and_classify(detector_test_image1)
    assert isinstance(result, tuple)


def test_composer_img2(detector_test_image2):
    result = composer.detect_and_classify(detector_test_image2)
    assert isinstance(result, tuple)


def test_composer_batch(detector_test_image1, detector_test_image2):
    result = composer.detect_and_classify_batch(
        [detector_test_image1, detector_test_image2]
    )
    assert len(result) == 2
