from pathlib import Path

from maddrive_adas.sign_det.detector import YoloV5Detector
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier
from maddrive_adas.sign_det.composer import BasicSignsDetectorAndClassifier
from maddrive_adas.sign_det.base import (
    AbstractSignDetector, AbstractSignClassifier,
    AbstractComposer, DetectedInstance
)

PROJECT_ROOT: Path = Path('.')
SIGN_DETECTOR_MODEL_ARCHIVE: str = str(
    PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'detector_config_img_size'
)
SIGN_CLASSIFIER_MODEL_ARCHIVE: str = str(
    PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'encoder_cl_config'
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


def test_detector_base_execution_img1(detector_test_image1):
    detections = detector.detect(detector_test_image1)
    assert len(detections.confs) == 3  # in fact 2


def test_detector_base_execution_img2(detector_test_image2):
    detections = detector.detect(detector_test_image2)
    assert len(detections.confs) == 3  # in fact 2


def test_detector_base_execution_batch(
    detector_test_image1,
    detector_test_image2
):
    detections = detector.detect_batch(
        [detector_test_image1, detector_test_image2])
    assert len(detections) == 2  # in fact 2


def test_detector_base_execution_batch_empty():
    detections = detector.detect_batch([])
    assert len(detections) == 0  #


def test_classifier_base_2_1_1(classifier_test_image_2_1_1):
    classification_res = classifier.classify(classifier_test_image_2_1_1)
    print(classification_res)
    assert classification_res  # classification_res[0] == '2.1'


def test_classifier_base_2_1_2(classifier_test_image_2_1_2):
    classification_res = classifier.classify(classifier_test_image_2_1_2)
    print(classification_res)
    assert classification_res  # classification_res[0] == '2.1'


def test_classifier_base_5_16(classifier_test_image_5_16: DetectedInstance):
    classification_res = classifier.classify(classifier_test_image_5_16)
    print(classification_res)
    assert classification_res  # classification_res[0] == '5.16'


def test_classifier_batch_test(
    classifier_test_image_2_1_1,
    classifier_test_image_2_1_2,
    classifier_test_image_5_16
):
    classification_res = classifier.classify_batch(
        [
            classifier_test_image_2_1_1,
            classifier_test_image_2_1_2,
            classifier_test_image_5_16
        ]
    )
    print(classification_res)
    assert classification_res  # [x[0] for x in classification_res] == ['2.1', '2.1', '5.16']


def test_classifier_batch_empty():
    classification_res = classifier.classify_batch([])
    print(classification_res)
    assert len(classification_res) == 0  # in fact 2


def test_composer_img1(detector_test_image1):
    result = composer.detect_and_classify(detector_test_image1)
    print(result)
    assert isinstance(result, tuple)


def test_composer_img2(detector_test_image2):
    result = composer.detect_and_classify(detector_test_image2)
    print(result)
    assert isinstance(result, tuple)


def test_composer_batch(detector_test_image1, detector_test_image2):
    result = composer.detect_and_classify_batch(
        [detector_test_image1, detector_test_image2]
    )
    print(result)
    assert len(result) == 2
