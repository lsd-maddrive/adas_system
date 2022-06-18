import numpy as np
import cv2


class DetectedInstance:  # TODO: remove detected sign class?
    """Describes instance for classifier.
    """

    def __init__(self, img: np.array):
        self.abs_rois: list[int] = []
        self.rel_rois: list[float] = []
        self.confs: list[float] = []
        self.img = img.copy()

    def add_abs_roi(self, roi: list[int], conf: float):
        self.abs_rois.append(list(map(int, roi)))
        w, h, *_ = self.img.shape
        self.confs.append(conf)
        self.rel_rois.append([
            roi[0] / h,
            roi[1] / w,
            roi[2] / h,
            roi[3] / w,
        ])

    def add_rel_roi(self, roi: list[int], conf: float):
        self.rel_rois.append(roi)
        w, h, *_ = self.img.shape
        self.confs.append(conf)
        self.abs_rois.append(list(map(int, [
            h * roi[0],
            w * roi[1],
            h * roi[2],
            w * roi[3],
        ])))

    def get_rel_roi(self, idx):  # TODO: ret confidence
        try:
            return self.rel_rois[idx]
        except IndexError:
            assert False, 'Wrong index'

    def get_abs_roi(self, idx):
        try:
            return self.abs_rois[idx]
        except IndexError:
            assert False, 'Wrong index'

    def show_img(self):
        """Show image with detections.
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

    def get_roi_coint(self) -> int:
        return len(self.rel_rois)

    def get_cropped_img(self, roi_idx) -> np.array:
        rroi = self.get_rel_roi(roi_idx)
        w, h, *_ = self.img.shape
        return self.img[
            int(rroi[0] * w): int(rroi[2] * w),
            int(rroi[1] * h): int(rroi[3] * h),
        ]


class DetectedSign:
    """Class to control detected sign operations.

    We define one point of signs retectino contract for communication
    """

    def __init__(self, bbox: list[float]) -> None:
        self._bbox = np.array(bbox, dtype=np.float32)

    def as_dict(self) -> dict:
        return {"bbox": self._bbox.tolist()}


class AbstractSignDetector:
    """Base Detector.
    """

    def __init__(*args, **kwargs) -> None:  # TODO: сделать общий конструктов?
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

    def __init__(*args, **kwargs) -> None:
        raise NotImplementedError()

    def classify_batch(
        self,
        imgs: list[np.array], RROI: list[list[float]]
    ) -> list[tuple[str, float]]:
        """Classify batch.

        Args:
            imgs (list[np.array]): List of images.
            RROI (list[list[float]]): List of relative regions of interest.

        Returns:
            list[tuple[str, float]]: List of results.
        """
        raise NotImplementedError()

    def classify(self, img: np.array, RROI: list[float]) -> list:
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

    # TODO: fix my name please
    def detect_and_classify_batch(self, imgs: list[np.array]) -> None:  # list[tuple[str, float]]:
        raise NotImplementedError()

    # TODO: fix my name please
    def detect_and_classify(self, imgs: list[np.array]) -> None:  # list[tuple[str, float]]:
        raise NotImplementedError()
