import cv2
from pathlib import Path
from .logger import logger


# def imread_rgb(fpath: str | Path):
def imread_rgb(fpath):
    if isinstance(fpath, Path):
        fpath = str(fpath)

    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    if img is None:
        logger.critical(
            f'{__name__} Unable to read img.'
        )
        assert False, f'Verify image path: {fpath}'
        return img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
