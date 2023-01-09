from typing import List, Tuple

import torch
import numpy as np

from .base import AbstractSignDetector, DetectedInstance
from maddrive_adas.utils.general import non_max_suppression, scale_coords
from maddrive_adas.utils.augmentations import letterbox
from maddrive_adas.utils.checkpoint import Checkpoint


class YoloV5Detector(AbstractSignDetector):
    """Traffic sign detector."""

    def __init__(
        self,
        model_archive_file_path: str,
        device: torch.device,
        iou_thres: float,
        conf_thres: float
    ):
        self._iou_thres = iou_thres
        self._conf_thres = conf_thres

        self._device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._checkpoint = Checkpoint(model_archive_file_path)
        self._img_size = self._checkpoint.get_checkpoint_img_size()
        self._img_size = (self._img_size, self._img_size)
        self._model = self._checkpoint.load_eval_checkpoint(map_device=self._device)

    def _transform_single_img(self, img: np.ndarray) -> torch.Tensor:
        """Transform single img to model input.

        Args:
            img (np.ndarray): Input RGB image.

        Returns:
            torch.Tensor: Input for model.
        """
        frame: np.ndarray = letterbox(img, self._img_size, auto=False)[0]    # resize + letterbox
        frame = frame.transpose((2, 0, 1))  # [::-1] skip BRG to RGB, coz input image should be RGB
        frame = np.ascontiguousarray(frame)     # idk what this, maybe memory trick
        frame = torch.from_numpy(frame).float()     # uint8 -> float
        frame /= 255        # to 0.0 - 1.0
        frame = frame[None, ...]    # add 4 dim
        return frame

    @staticmethod
    def translatePreds(
        pred,
        nn_img_size,
        source_img_size,
        conf_thres,
        iou_thres,
        max_det,
    ) -> List[dict]:    # TODO: fix annotation

        pred = non_max_suppression(
            pred,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det
        )

        ret_list: List[DetectedInstance] = []

        for detections, img_size in zip(pred, source_img_size):
            detect_info: List[Tuple[List[int], float]] = []
            if len(detections):
                detections[:, :4] = scale_coords(
                    nn_img_size, detections[:, :4], img_size
                ).round()

                detect_info.extend(
                    detections[:, :5].tolist()
                )

            ret_list.append(detect_info)

        return ret_list

    @torch.no_grad()
    def detect_batch(
        self,
        imgs: List[np.ndarray],
    ) -> List[DetectedInstance]:
        """Returs list of subimages - detected signs.

        Return list is list of detected signs per each imgs element.

        Args:
            imgs (List[np.ndarray]): List of RGB images.

        Returns:
            List[DetectedInstance]: DetectedInstance image description.
        """
        if not imgs:
            return []

        original_img_size: List[int] = []
        for img in imgs:
            original_img_size.append((img.shape[0], img.shape[1]))

        # transform to list to batch
        # TODO: this might be slow, use torch.tensor in future
        transformed_imgs = [self._transform_single_img(img) for img in imgs]
        batch = torch.cat(transformed_imgs, dim=0).to(self._device)
        preds = self._model(batch)[0]   # why 0? models.common:398 DetectMultiBackend
        # i realy dont know what model output contains besides coords

        translated_preds = YoloV5Detector.translatePreds(
            preds,
            self._img_size,  # scaled img for model
            original_img_size,  #
            conf_thres=self._conf_thres,
            iou_thres=self._iou_thres,
            max_det=20)

        # transform to DetectedInstance
        ret_list: List[DetectedInstance] = []
        assert len(translated_preds) == len(imgs), 'Array len mismatch'
        for img, tpreds in zip(imgs, translated_preds):
            di = DetectedInstance(img)
            for pred in tpreds:
                di.add_abs_roi(pred[:4], pred[4])
            ret_list.append(di)

        return ret_list

    def detect(self, img: np.ndarray) -> DetectedInstance:
        """Detect sign on img.

        Args:
            img (np.ndarray): Input image.

        Returns:
            List[Tuple[float, float, float, float]]: List of relative sign coordinates.
        """
        return self.detect_batch([img])[0]
