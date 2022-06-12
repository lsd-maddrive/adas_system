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

    def __init__(self) -> None:  # TODO: сделать общий конструктов?
        raise NotImplementedError()

    def detect(self, img: np.array) -> dict[str, list]:
        """Detect signs on image list.

        Args:
            imgs (np.array): np.array images.

        Returns:
            list[dict]: dicts per image, that contains abs. coords, relative coords and confidence.
        """
        raise NotImplementedError()

    def detect_batch(self, imgs: list[np.array]) -> list[dict[str, list]]:
        """Detect signs on image list.

        Args:
            imgs (list[np.array]): list of np.array images.

        Returns:
            list[dict]: dicts per image, that contains abs. coords, relative coords and confidence.
        """
        raise NotImplementedError()


class AbstractSignClassifier:
    """Base Classifier.
    """

    def __init__(self) -> None:
        raise NotImplementedError()

    def classify_batch(
        self,
        imgs: list[np.array], relative_sign_pos: list[list[float]]
    ) -> list[tuple[str, float]]:
        raise NotImplementedError()

    def classify(self, img: np.array, relative_sign_pos: list[float]) -> list:
        raise NotImplementedError()


class AbstractComposer:
    """Composes AbstatractSignsDetector & AbstatractSignsClassifier.
    """

    def __init__(self, detector: AbstatractSignDetector, classifier: AbstractSignClassifier):
        """Composes Detector and Classifiner.

        Args:
            detector (AbstatractSignDetector): Initialized Detector.
            classifier (AbstractSignClassifier): Initialized Classifier.
        """
        raise NotImplementedError()

    # TODO: fix my name please
    def detect_and_classify_batch(self, imgs: list[np.array]) -> None:  # list[tuple[str, float]]:
        raise NotImplementedError()

    # TODO: fix my name please
    def detect_and_classify(self, imgs: list[np.array]) -> None:  # list[tuple[str, float]]:
        raise NotImplementedError()
