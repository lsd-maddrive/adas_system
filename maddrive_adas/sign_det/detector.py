import copy
from pathlib import Path
import torch
import numpy as np

from base import AbstatractSignDetector
from src.utils.general import non_max_suppression, scale_coords
from src.models.yolo import Model
from src.utils.augmentations import letterbox
from src.utils.fs import imread_rgb

DETECT_INFO_PROTO = {
    'relative_coords': [],
    'confs': [],
    'coords': []
}

REQUIRED_ARCHIVE_KEYS = ['model', 'input_image_size', 'model_config']


class YoloV5Detector(AbstatractSignDetector):

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
    ) -> list[dict]:

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

        for i, det in enumerate(pred):
            detect_info = copy.deepcopy(DETECT_INFO_PROTO)
            if len(det):
                current_img_size = source_img_size[i]
                det[:, :4] = scale_coords(
                    nn_img_size, det[:, :4], current_img_size
                ).round()
                for det_instance in det:
                    det_instance_f = det_instance[:4].float()
                    detect_info['coords'].append(
                        list(map(int, det_instance_f)))
                    detect_info['relative_coords'].append(list(map(float, [
                        det_instance_f[0] / current_img_size[1],
                        det_instance_f[1] / current_img_size[0],
                        det_instance_f[2] / current_img_size[1],
                        det_instance_f[3] / current_img_size[0],
                    ])))
                detect_info['confs'].append(list(map(float, det[:, 4])))

            ret_list.append(detect_info)

        return ret_list

    @torch.no_grad()
    def detect_batch(self, imgs: list[np.array]) -> list[list[np.array]]:
        """Returs list of subimages - detected signs.

        Return list is list of detected signs per each imgs element.

        Args:
            imgs (List[np.array]): List of RGB images.

        Returns:
            List[np.array]: RGB Subimages list for every batch element.
        """
        if len(imgs) == 0:
            return []

        original_img_size: list[int] = []
        for img in imgs:
            original_img_size.append((img.shape[0], img.shape[1]))

        # transform to list to batch
        # TODO: this might be slow, use torch.tensor in future
        transformed_imgs = [self._transform_single_img(img) for img in imgs]
        batch = torch.cat(transformed_imgs, dim=0).to(self._device)
        preds = self._model(batch)[0]   # why 0? models.common:398 DetectMultiBackend
        # i realy dont know what model output contains besides coords

        data = YoloV5Detector.translatePreds(
            preds,
            self._img_size,  # scaled img for model
            original_img_size,  #
            # TODO: cardcoded arg
            conf_thres=0.11,
            max_det=10)

        return data

    def detect(self, img: np.array) -> list[tuple[float, float, float, float]]:
        """Detect sign on img.

        Args:
            img (np.array): Input image.

        Returns:
            list[tuple[float, float, float, float]]: List of relative sign coordinates.
        """
        return self.detect_batch([img])[0]


def test():
    PROJECT_ROOT = Path('.')
    DATA_DIR = PROJECT_ROOT / 'tests' / 'test_data'
    MODEL_ARCHIVE = PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'detector_config_img_size'

    c: AbstatractSignDetector = YoloV5Detector(config_path=str(MODEL_ARCHIVE))

    img1 = imread_rgb(DATA_DIR / 'custom_test.png')
    img2 = imread_rgb(DATA_DIR / 'test_image.png')

    sign = c.detect_batch([img1, img2])
    # sign = c.detect(img2)

    return sign


if __name__ == '__main__':
    test()
