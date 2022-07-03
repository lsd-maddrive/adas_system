import cv2
from pathlib import Path
from .logger import logger

import hashlib


# def imread_rgb(fpath: str | Path):
def imread_rgb(fpath):
    if isinstance(fpath, Path):
        fpath = str(fpath)

    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    if img is None:
        logger.critical(
            f'{__name__} Unable to read img.'
        )
        raise ValueError(f'Verify image path: {fpath}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_file_sha256(fpath):
    hash = hashlib.sha256()
    with open(fpath,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            hash.update(byte_block)

    hex_value = hash.hexdigest()
    return hex_value


def get_file_md5(fpath):
    hash = hashlib.md5()
    with open(fpath,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            hash.update(byte_block)

    hex_value = hash.hexdigest()
    return hex_value
