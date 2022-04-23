"""YoloV5 based detector."""

from typing import List, Tuple
import torch
import numpy as np

from src.utils.logger import logger
from src.models.yolo import Model
from src.utils.augmentations import letterbox


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

            self._device = device

            self._model = Model(cfg=path_to_cfg, ch=3, nc=1)
            self._model.load_state_dict(
                torch.load(path_to_weights),
            )

            self._model.half() if use_half and device.type != 'cpu' else self._model.float()
            self._model.to(self._device)

            self._is_initialized = True

        except (FileNotFoundError, KeyError) as exc_obj:
            logger.info(
                f'{__name__} Cannot initalized backend: {exc_obj}.'
            )

    def _transform_single_img(self, img) -> np.array:
        """TODO: _summary_

        Args:
            img (_type_): _description_

        Returns:
            np.array[int]: _description_
        """
        frame = letterbox(img, self.img_size, auto=False)[0]
        frame = frame.transpose((2, 0, 1))[::-1]
        frame = np.ascontiguousarray(frame)
        frame = torch.from_numpy(frame).float()
        frame /= 255
        frame = frame[None, ...]
        return frame

    def detect_batch(self, imgs: List[np.array]) -> List[np.array]:
        """_summary_

        Args:
            imgs (List[np.array]): List of HWD.

        Returns:
            List[np.array]: _description_
        """
