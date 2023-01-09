from pathlib import Path

import pytest

from maddrive_adas.utils import fs
from maddrive_adas.utils import get_project_root
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier
from maddrive_adas.sign_det.base import AbstractSignClassifier, DetectedInstance

PROJECT_ROOT = get_project_root()
SIGN_CLASSIFIER_MODEL_ARCHIVE = str(PROJECT_ROOT / 'classifier_archive')
classifier: AbstractSignClassifier = EncoderBasedClassifier(
    config_path=SIGN_CLASSIFIER_MODEL_ARCHIVE
)


@pytest.fixture
def test_data_path():
    test_data_path = Path(__file__).parent / "test_data" / "test_detector_classifier"
    return test_data_path


@pytest.fixture
def classifier_test_img_2_1_2(test_data_path: Path):
    img_fpath = test_data_path / '2.1_2.png'
    img = fs.imread_rgb(img_fpath)
    return img


@pytest.fixture
def classifier_test_di_2_1_2(classifier_test_img_2_1_2):
    d = DetectedInstance(classifier_test_img_2_1_2)
    d.add_rel_roi([0., 0., 1., 1.], 1.)
    return d


@pytest.fixture
def classifier_test_img_5_19(test_data_path: Path):
    img_fpath = test_data_path / '5.19.png'
    img = fs.imread_rgb(img_fpath)
    return img


@pytest.fixture
def classifier_test_di_5_19(classifier_test_img_5_19):
    d = DetectedInstance(classifier_test_img_5_19)
    d.add_rel_roi([0., 0., 1., 1.], 1.)
    return d


@pytest.fixture
def classifier_test_img_7_4(test_data_path: Path):
    img_fpath = test_data_path / '7.4.png'
    img = fs.imread_rgb(img_fpath)
    return img


@pytest.fixture
def classifier_test_di_7_4(classifier_test_img_7_4) -> DetectedInstance:
    d = DetectedInstance(classifier_test_img_7_4)
    d.add_rel_roi([0., 0., 1., 1.], 1.)
    return d


@pytest.mark.classifier
def test_classifier_base_2_1_1_img(classifier_test_img_2_1_2):
    classification_res = classifier.classify(classifier_test_img_2_1_2)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


@pytest.mark.classifier
def test_classifier_base_2_1_2_di(classifier_test_di_2_1_2):
    classification_res = classifier.classify(classifier_test_di_2_1_2)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


@pytest.mark.classifier
def test_classifier_base_5_19_img(classifier_test_img_5_19):
    classification_res = classifier.classify(classifier_test_img_5_19)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


@pytest.mark.classifier
def test_classifier_base_5_19_di(classifier_test_di_5_19):
    classification_res = classifier.classify(classifier_test_di_5_19)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


@pytest.mark.classifier
def test_classifier_base_7_4_img(classifier_test_img_7_4):
    classification_res = classifier.classify(classifier_test_img_7_4)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


@pytest.mark.classifier
def test_classifier_base_7_4_di(classifier_test_di_7_4):
    classification_res = classifier.classify(classifier_test_di_7_4)
    assert isinstance(classification_res, tuple) and len(classification_res[1]) == 1


@pytest.mark.classifier
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


@pytest.mark.classifier
def test_classifier_batch_empty():
    classification_res = classifier.classify_batch([])
    assert len(classification_res) == 0
