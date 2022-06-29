from typing import List, Tuple

import numpy as np
import cv2


class DetectedInstance:  # TODO: remove detected sign class?
    """Describes instance for classifier.
    """

    def __init__(self, img: np.array):
        self.abs_rois: List[int] = []
        self.rel_rois: List[float] = []
        self.confs: List[float] = []
        self.img = img.copy()

    def add_abs_roi(self, roi: List[int], conf: float):
        self.abs_rois.append(list(map(int, roi)))
        h, w, *_ = self.img.shape
        self.confs.append(conf)
        self.rel_rois.append([
            roi[0] / w,
            roi[1] / h,
            roi[2] / w,
            roi[3] / h,
        ])

    def add_rel_roi(self, roi: List[int], conf: float):
        self.rel_rois.append(roi)
        h, w, *_ = self.img.shape
        self.confs.append(conf)
        self.abs_rois.append(list(map(int, [
            w * roi[0],
            h * roi[1],
            w * roi[2],
            h * roi[3],
        ])))

    def get_rel_roi(self, idx):
        """Get relative ROI and confidence.
        """
        try:
            return self.rel_rois[idx], self.confs[idx]
        except IndexError:
            assert False, 'Wrong index'

    def get_abs_roi(self, idx) -> Tuple[list, float]:
        """Get absolute ROI and confidence.
        """
        try:
            return self.abs_rois[idx], self.confs[idx]
        except IndexError:
            assert False, 'Wrong index'

    def _show_img(self):
        """Show image with detections.

        This method requires development dependencies or not
        headless OpenCV2
        """
        img_ = self.img.copy()
        for idx, abs_roi in enumerate(self.abs_rois):
            img_ = cv2.rectangle(
                img_,
                (abs_roi[0], abs_roi[1]),
                (abs_roi[2], abs_roi[3]),
                (0, 0, 255), 2
            )
            img_ = cv2.putText(
                img_,
                str(idx), (abs_roi[0], abs_roi[3]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )
        cv2.imshow(f'{self}', img_)
        cv2.waitKey(3)

    def get_roi_count(self) -> int:
        """Get ROI count.

        Returns:
            int: ROI count.
        """
        return len(self.rel_rois)

    def get_cropped_img(self, roi_idx: int) -> np.array:
        """Get cropped ROID image.

        Args:
            roi_idx (int): index of ROI.

        Returns:
            np.array: cropped image.
        """
        return self.img[
            self.abs_rois[roi_idx][1]: self.abs_rois[roi_idx][3],
            self.abs_rois[roi_idx][0]: self.abs_rois[roi_idx][2]
        ]


class DetectedSign:
    """Class to control detected sign operations.

    We define one point of signs retectino contract for communication
    """

    def __init__(self, bbox: List[float]) -> None:
        self._bbox = np.array(bbox, dtype=np.float32)

    def as_dict(self) -> dict:
        return {"bbox": self._bbox.tolist()}


class AbstractSignDetector:
    """Base Detector.
    """

    def __init__(*args, **kwargs) -> None:  # TODO: сделать общий конструктор?
        raise NotImplementedError()

    def detect(self, img: np.array) -> DetectedInstance:
        """Detect signs on image list.

        Args:
            imgs (np.array): np.array images.

        Returns:
            DetectedInstance: Detected signs as DetectedInstance.
        """
        raise NotImplementedError()

    def detect_batch(self, imgs: List[np.array]) -> List[DetectedInstance]:
        """Detect signs on image list.

        Args:
            imgs (List[np.array]): list of np.array images.

        Returns:
            List[DetectedInstance]: List of Detected signs as DetectedInstance's.
        """
        raise NotImplementedError()


class AbstractSignClassifier:
    """Base Classifier.
    """

    def __init__(*args, **kwargs) -> None:
        raise NotImplementedError()

    def classify_batch(
        self,
        instances: List[DetectedInstance]
    ) -> List[Tuple[str, float]]:
        """Classify batch.

        Args:
            imgs (List[np.array]): List of images.
            RROI (List[List[float]]): List of relative regions of interest.

        Returns:
            List[Tuple[str, float]]: Unzipped list of results: (sign, confidence).
        """
        raise NotImplementedError()

    def classify(self, instance: DetectedInstance) -> Tuple[str, float]:
        raise NotImplementedError()


class AbstractComposer:
    """Composes AbstatractSignsDetector & AbstatractSignsClassifier.
    """

    # TODO: allow to use only this constructor
    def __init__(self, detector: AbstractSignDetector, classifier: AbstractSignClassifier):
        """Composes Detector and Classifier.

        Args:
            detector (AbstatractSignDetector): Initialized Detector.
            classifier (AbstractSignClassifier): Initialized Classifier.
        """
        raise NotImplementedError()

    # TODO: fix my name and return types please
    def detect_and_classify_batch(self, imgs: List[np.array]) -> None:  # List[Tuple[str, float]]:
        raise NotImplementedError()

    # TODO: same
    def detect_and_classify(self, imgs: List[np.array]) -> None:  # List[Tuple[str, float]]:
        raise NotImplementedError()
