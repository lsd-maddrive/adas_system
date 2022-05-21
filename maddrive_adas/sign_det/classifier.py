import numpy as np
from typing import List

from base import BaseSignsClassifier

import json
import torch
from torch import nn
from src.utils.models import get_model_and_img_size
from src.utils.fs import imread_rgb
from src.utils.checkpoint import load_checkpoint
from src.utils.transforms import get_minimal_and_augment_transforms
# TODO:
# move constructor to BaseSignsClassifier class aka super init


class NonEncoderBasedClassifier(BaseSignsClassifier):
    def __init__(
        self,
        path_to_weights: str,
        path_to_model_config: str,
        path_to_centroid_location: str,
        device: torch.device,

    ):

        self._model: nn.Module
        self._model, self._img_size = get_model_and_img_size(
            path_to_config=path_to_model_config
        )
        self._model.to(device)

        self._transform, _ = get_minimal_and_augment_transforms(
            self._img_size
        )
        self._model, _, _, _, _ = load_checkpoint(
            self._model,
            None,
            None,
            path_to_weights)

        with open(path_to_centroid_location, 'r') as f:
            _centroid_location: dict = json.load(f)
            _centroid_location_dict: dict = \
                {k: torch.Tensor(v) for k, v in _centroid_location.items()}

        self._idx_to_key: list = {idx: v for idx, v in _centroid_location_dict.items()}
        self._centroid_location: torch.Tensor = _centroid_location_dict.values()

    # maybe ret list[tuple(SIGN, CONFIDENCE: float)] ?
    def classify(self, imgs: List[np.array]) -> List[np.array]:
        res = []
        res = self._model(imgs)
        res1 = self._get_nearest_centroids(res)
        return res1

    def _get_nearest_centroids(self, embs) -> str:
        dist = (embs - self._centroid_location).pow(2).sum(-1).sqrt()
        key = self._idx_to_key[int(torch.argmin(dist))]
        return key


def test():
    device = torch.device('cuda:0')
    c = NonEncoderBasedClassifier(
        path_to_weights='EXCLUDE_ADDI_SIGNSencoder_loss_1e-05_acc_0.99566epoch_99.encoder',
        path_to_model_config='encoder_config.json',
        path_to_centroid_location='centroid_location.txt',
        device=device
    )

    imread_rgb()

    return c


if __name__ == "__main__":
    test()
