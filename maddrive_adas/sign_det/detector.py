from typing import List, Tuple

import torch
import numpy as np

from .base import AbstractSignDetector, DetectedInstance
from maddrive_adas.utils.general import non_max_suppression, scale_coords
from maddrive_adas.models.yolo import Model
from maddrive_adas.utils.augmentations import letterbox

REQUIRED_ARCHIVE_KEYS = ['model', 'input_image_size', 'model_config']


class YoloV5Detector(AbstractSignDetector):

    def __init__(
        self,
        config_path: str,
        device: torch.device = None
    ):
        """Detector Constructor.

        Args:
            config_path (str): path to archive with REQUIRED_ARCHIVE_KEYS.
            device (torch.device, optional): specific torch.device. Defaults to None.
        """
        self._device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        model_dict: dict = torch.load(config_path)
        assert(all([key in model_dict.keys() for key in REQUIRED_ARCHIVE_KEYS])
               ), f'Verify model archive keys. It should contain {REQUIRED_ARCHIVE_KEYS}'

        self._img_size = (
            model_dict['input_image_size'],
            model_dict['input_image_size']
        )

        self._model = Model(
            cfg=dict(model_dict['model_config']),
            ch=3,
            nc=1
        )

        self._model.load_state_dict(
            model_dict['model']
        )

        # Do not forget to eval after weights loaded, lmao
        self._model.eval()
        self._model.to(self._device)

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

    # TODO: hardcoded args, move it to model config archive-dict
    @staticmethod
    def translatePreds(
        pred,
        nn_img_size,
        source_img_size,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
    ) -> List[dict]:    # TODO: fix annotation

        pred = non_max_suppression(
            pred,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic,
            multi_label=multi_label,
            labels=labels,
            max_det=max_det,
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
    def detect_batch(self, imgs: List[np.array]) -> List[DetectedInstance]:
        """Returs list of subimages - detected signs.

        Return list is list of detected signs per each imgs element.

        Args:
            imgs (List[np.array]): List of RGB images.

        Returns:
            List[np.array]: RGB Subimages list for every batch element.
        """
        if len(imgs) == 0:
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
            # TODO: cardcoded arg
            conf_thres=0.10,
            max_det=10)

        # transform to DetectedInstance
        ret_list: List[DetectedInstance] = []
        assert len(translated_preds) == len(imgs), 'Array len mismatch'
        for img, tpreds in zip(imgs, translated_preds):
            di = DetectedInstance(img)
            for pred in tpreds:
                di.add_abs_roi(pred[:4], pred[4])
            ret_list.append(di)

        return ret_list

    def detect(self, img: np.array) -> DetectedInstance:
        """Detect sign on img.

        Args:
            img (np.array): Input image.

        Returns:
            List[Tuple[float, float, float, float]]: List of relative sign coordinates.
        """
        return self.detect_batch([img])[0]
