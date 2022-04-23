from base import BaseSignsClassifier

import torch
from torchvision.models import resnet18
from torch import nn

# TODO:
# move constructor to BaseSignsClassifier class aka super init


class EncoderBasedClassifier(BaseSignsClassifier):
    def __init__(
        self,
        path_to_weights: str,
        output_features: int,
        device: torch.device,

    ):
        self._model: resnet18 = resnet18(pretrained=True)
        self._model.fc = nn.Linear(
            in_features=512,
            out_features=output_features,
            bias=True)

        self._model.load_state_dict(torch.load(path_to_weights))
        self._model.to(device)


class NonEncoderBasedClassifier(BaseSignsClassifier):
    def __init__(
        self,
        path_to_weights: str,
        output_features: int,
        device: torch.device,

    ):
        self._model: nn.Module = resnet18(pretrained=True)
        self._model.fc = nn.Sequential(
            nn.Linear(
                in_features=512,
                out_features=output_features,
                bias=True
            )
        )

        self._model.load_state_dict(torch.load(path_to_weights))
        self._model.to(device)
