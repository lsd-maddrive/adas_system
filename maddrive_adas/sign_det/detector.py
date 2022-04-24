"""YoloV5 based detector."""


from typing import List, Tuple
import copy

import torch
import numpy as np

from src.utils.general import non_max_suppression, scale_coords
from src.utils.logger import logger
from src.models.yolo import Model
from src.utils.augmentations import letterbox

DETECT_INFO_PROTO = {
    "coords": [],
    "relative_coords": [],
    "class": [],
    "confs": [],
    "count": 0,
}


class YoloV5Detector():

    def __init__(
        self,
        path_to_cfg: str,
        path_to_weights: str,
        device: torch.device,
        img_size: Tuple[int, int] = (640, 640),
        use_half: bool = False
    ) -> bool:
        """_summary_
        TODO:

        Args:
            path_to_cfg (str): _description_
            path_to_weights (str): _description_
            device (torch.device): _description_
            use_half (bool, optional): _description_. Defaults to False.

        Returns:
            bool: _description_
        """
        try:
            if isinstance(img_size, int):
                img_size = (img_size, img_size)

            self.img_size = img_size
            self._device = device

            self._model = Model(cfg=path_to_cfg, ch=3, nc=1)
            self._model.load_state_dict(
                torch.load(path_to_weights),
            )

            # Do not forget to eval after weights loaded, lmao
            self._model.eval()

            if use_half and device.type != 'cpu':
                self._model.half()
                self._im_half = True
            else:
                self._model.float()
                self._im_half = False

            self._model.half() if use_half and device.type != 'cpu' else self._model.float()
            self._model.to(self._device)

            self._is_initialized = True

        except (FileNotFoundError, KeyError) as exc_obj:
            logger.info(
                f'{__name__} Cannot initalized backend: {exc_obj}.'
            )
            self._is_initialized = False

    def _transform_single_img(self, img: np.ndarray) -> torch.Tensor:
        """_summary_

        Args:
            img (np.ndarray): Input RGB image.

        Returns:
            torch.Tensor: Input for model.
        """
        frame: np.ndarray = letterbox(img, self.img_size, auto=False)[0]    # resize + letterbox
        frame = frame.transpose((2, 0, 1))  # [::-1] skip BRG to RGB, coz input image should be RGB
        frame = np.ascontiguousarray(frame)     # idk what this, maybe memory trick
        frame = torch.from_numpy(frame).float()     # uint8 -> float
        frame /= 255        # to 0.0 - 1.0
        frame = frame[None, ...]    # add 4 dim
        return frame

    # TODO: hardcoded args
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
    ) -> List[dict]:

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

        ret_list: list = []

        # TODO:
        # make it parrallel aka multithreaded
        for i, det in enumerate(pred):
            detect_info = copy.deepcopy(DETECT_INFO_PROTO)
            if len(det):
                detect_info["relative_coords"].append(det[:, :4])
                det[:, :4] = scale_coords(
                    nn_img_size, det[:, :4], source_img_size[i]
                ).round()

                for *xyxy, conf, cls in reversed(det):
                    detect_info["coords"].append(list(map(int, xyxy)))
                    detect_info["confs"].append(float(conf))
                    detect_info["class"].append(int(cls))

                    detect_info["count"] += 1

            ret_list.append(detect_info)

        return ret_list

    @torch.no_grad()
    def detect_batch(self, imgs: List[np.array]) -> List[List[np.array]]:
        """Returs list of subimages - detected signs.

        Return list is list of detected signs per each imgs element.

        Args:
            imgs (List[np.array]): List of RGB images.

        Returns:
            List[np.array]: RGB Subimages list for every batch element.
        """
        if not self._is_initialized:
            logger.info(
                f'{__name__} Attempt to detect, but not initialized.'
            )
            return []

        if len(imgs) == 0:
            return []

        original_img_size: Tuple[int, int] = []
        for img in imgs:
            original_img_size.append((img.shape[0], img.shape[1]))

        # transform to list to batch
        # TODO: this might be slow, use torch.tensor in future
        transformed_imgs = [self._transform_single_img(img) for img in imgs]
        batch = torch.cat(transformed_imgs, dim=0)
        if self._im_half:
            batch = batch.half()

        batch = batch.to(self._device)

        preds = self._model(batch)[0]   # why 0? models.common:398 DetectMultiBackend
        # i realy dont know what model output contains besides coords

        data = self.translatePreds(
            preds,
            self.img_size,  # scaled img for model
            original_img_size,  #
            # TODO: cardcoded arg
            conf_thres=0.101,
            max_det=10)

        # data stores list (dict of detected signs) per imgs

        ret_list: List = []

        for idx, img_data in enumerate(data):
            per_image: List = []
            for i in range(img_data['count']):
                cropped = imgs[idx][
                    img_data['coords'][i][1]: img_data['coords'][i][3],
                    img_data['coords'][i][0]: img_data['coords'][i][2]
                ]
                per_image.append(cropped)

            ret_list.append(per_image)

        return ret_list
