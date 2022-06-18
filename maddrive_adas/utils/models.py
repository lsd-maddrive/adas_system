import json
import importlib

import torch
from maddrive_adas.utils.general import non_max_suppression, scale_coords
from maddrive_adas.utils.augmentations import (
    Albumentations,
    augment_hsv,
    letterbox,
    random_perspective,
)
from maddrive_adas.models.common import Conv
from maddrive_adas.models.yolo import Detect, Model
from maddrive_adas.models.experimental import Ensemble


def get_model_config(path_to_config: str):
    with open(path_to_config, 'rb') as f:
        config_data = f.read().decode('utf-8')
    model_data = json.loads(config_data)
    return model_data


def get_model_and_img_size(
    path_to_config: str = '',
    config_data: str = ''
) -> tuple[torch.nn.Module, int]:
    assert(path_to_config or config_data), f'{__name__} Unable to get model: empty args.'
    if path_to_config:
        model_data = get_model_config(path_to_config)
    else:
        model_data = json.loads(config_data)

    model = getattr(
        importlib.import_module('torchvision.models'),
        f'{model_data["base"]}')(pretrained=True)
    if 'efficientnet' in model_data["base"]:
        model.classifier[1] = torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=model_data['output_len'],
            bias=model_data['bias'])
    elif 'resnet' in model_data["base"]:
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features,
            out_features=model_data['output_len'],
            bias=model_data['bias'])
    img_size = model_data['input_image_size']
    return model, img_size


class makeDetectFromModel(torch.nn.Module):
    def __init__(self, model, device=None):
        super().__init__()

        ensemble = Ensemble()
        ensemble.append(model).float().eval()

        for m in model.modules():
            if type(m) in [
                torch.nn.Hardswish,
                torch.nn.LeakyReLU,
                torch.nn.ReLU,
                torch.nn.ReLU6,
                torch.nn.SiLU,
                Detect,
                Model,
            ]:
                m.inplace = True  # pytorch 1.7.0 compatibility
                if type(m) is Detect:
                    if not isinstance(
                        m.anchor_grid, list
                    ):  # new Detect Layer compatibility
                        delattr(m, "anchor_grid")
                        setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        self.model = model
        self.device = device

    def forward(self, im):
        return self.model(im)[0]

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        if (
            isinstance(self.device, torch.device) and self.device.type != "cpu"
        ):  # only warmup GPU models
            im = (
                torch.zeros(*imgsz)
                .to(self.device)
                .type(torch.half if half else torch.float)
            )  # input image
            self.forward(im)  # warmup

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
    ):

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

        ret_dict = {
            "coords": [],
            "relative_coords": [],
            "class": [],
            "confs": [],
            "count": 0,
        }

        for i, det in enumerate(pred):
            if len(det):
                ret_dict["relative_coords"].append(det[:, :4])
                det[:, :4] = scale_coords(
                    nn_img_size, det[:, :4], source_img_size
                ).round()

                for *xyxy, conf, cls in reversed(det):
                    ret_dict["coords"].append(list(map(int, xyxy)))
                    ret_dict["confs"].append(float(conf))
                    ret_dict["class"].append(int(cls))

                    ret_dict["count"] += 1

        return ret_dict
