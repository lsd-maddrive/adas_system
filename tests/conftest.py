from pathlib import Path

import pytest
import numpy as np

from maddrive_adas.utils import fs
from maddrive_adas.sign_det.base import DetectedInstance

# Path


@pytest.fixture
def test_data_dpath():
    test_data_dpath = Path(__file__).parent / "test_data"
    return test_data_dpath

# Detector


@pytest.fixture
def detector_test_image1(test_data_dpath: Path) -> np.ndarray:
    img_fpath = test_data_dpath / 'test_image.png'
    img = fs.imread_rgb(img_fpath)
    return img


@pytest.fixture
def detector_test_image2(test_data_dpath: Path) -> np.ndarray:
    img_fpath = test_data_dpath / 'custom_test.png'
    img = fs.imread_rgb(img_fpath)
    return img

# Classifier img and DetectedInstance's


@pytest.fixture
def classifier_test_img_2_1_2(test_data_dpath: Path) -> np.ndarray:
    img_fpath = test_data_dpath / '2.1_2.png'
    img = fs.imread_rgb(img_fpath)
    return img


@pytest.fixture
def classifier_test_di_2_1_2(classifier_test_img_2_1_2) -> DetectedInstance:
    d = DetectedInstance(classifier_test_img_2_1_2)
    d.add_rel_roi([0., 0., 1., 1.], 1.)
    return d


@pytest.fixture
def classifier_test_img_5_19(test_data_dpath: Path) -> np.ndarray:
    img_fpath = test_data_dpath / '5.19.png'
    img = fs.imread_rgb(img_fpath)
    return img


@pytest.fixture
def classifier_test_di_5_19(classifier_test_img_5_19) -> DetectedInstance:
    d = DetectedInstance(classifier_test_img_5_19)
    d.add_rel_roi([0., 0., 1., 1.], 1.)
    return d


@pytest.fixture
def classifier_test_img_7_4(test_data_dpath: Path) -> np.ndarray:
    img_fpath = test_data_dpath / '7.4.png'
    img = fs.imread_rgb(img_fpath)
    return img


@pytest.fixture
def classifier_test_di_7_4(classifier_test_img_7_4) -> DetectedInstance:
    d = DetectedInstance(classifier_test_img_7_4)
    d.add_rel_roi([0., 0., 1., 1.], 1.)
    return d
