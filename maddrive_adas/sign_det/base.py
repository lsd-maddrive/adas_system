import numpy as np


class DetectedSign:
    """Class to control detected sign operations.

    We define one point of signs retectino contract for communication
    """

    def __init__(self, bbox: list[float]) -> None:
        self._bbox = np.array(bbox, dtype=np.float32)

    def as_dict(self) -> dict:
        return {"bbox": self._bbox.tolist()}


class AbstatractSignDetector:
    """Base Detector.
    """

    def __init__(self) -> None:
        raise NotImplementedError()

    def detect(self, img: np.array) -> dict:
        raise NotImplementedError()

    def detect_batch(self, imgs: list[np.array]) -> list[dict]:
        raise NotImplementedError()


class AbstractSignClassifier:
    """Base Classifier.
    """

    def __init__(self) -> None:
        raise NotImplementedError()

    def classify(self, imgs: list[np.array]) -> list[tuple[str, float]]:
        raise NotImplementedError()


class AbstractComposer:
    """Composes AbstatractSignsDetector & AbstatractSignsClassifier.
    """

    def __init__(self) -> None:
        raise NotImplementedError()

    # TODO: fix my name please
    def detect_and_classify_batch(self, imgs: list[np.array]) -> None:  # list[tuple[str, float]]:
        raise NotImplementedError()

    # TODO: fix my name please
    def detect_and_classify(self, imgs: list[np.array]) -> None:  # list[tuple[str, float]]:
        raise NotImplementedError()
