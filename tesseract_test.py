import subprocess
from pathlib import Path

from maddrive_adas.sign_det.base import AbstractSignClassifier
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier
from maddrive_adas.utils.fs import imread_rgb

try:
    output = subprocess.check_output(
        'tesseract -v',
        stderr=subprocess.STDOUT
    ).decode()
    if 'tesseract' not in output:
        raise subprocess.CalledProcessError
except subprocess.CalledProcessError:
    print('Unable to call tessecact. Install and add tesseract to PATH variable.')
    print('Link: https://tesseract-ocr.github.io/tessdoc/Downloads.html')
    raise subprocess.CalledProcessError

PROJECT_ROOT = Path('.')
CLASSIFIER_ARCHIVE = PROJECT_ROOT / 'encoder_archive'

c: AbstractSignClassifier = EncoderBasedClassifier(
    config_path=str(CLASSIFIER_ARCHIVE)
)


data_path: Path = Path("D:\\d_tsw\\main_diplom\\SignDetectorAndClassifier\\data\\additional_sign")
for file in data_path.iterdir():
    if '3.25' in str(file):  # or '3.25' in str(file):
        print(file)
        img = imread_rgb(file)
        # res = c.classify(img)
        c._fixup_signs_with_text(img, ('3.25', 1))
assert False, 'Check windows'
