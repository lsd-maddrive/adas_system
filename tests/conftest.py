import os

import pytest


@pytest.fixture
def test_data_dpath():
    current_dpath = os.path.abspath(os.path.join(__file__))
    test_data_dpath = os.path.join(current_dpath, "test_data")
    return test_data_dpath
