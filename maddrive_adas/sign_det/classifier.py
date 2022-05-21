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
        self._device = device
        self._model: nn.Module
        self._model, self._img_size = get_model_and_img_size(
            path_to_config=path_to_model_config
        )
        self._model.to(device)

        self._transform, _ = get_minimal_and_augment_transforms(
            self._img_size
        )
        self._model, _, _, _ = load_checkpoint(
            self._model,
            None,
            None,
            path_to_weights)

        with open(path_to_centroid_location, 'r') as f:
            _centroid_location: dict = json.load(f)
            _centroid_location_dict: dict = \
                {k: torch.Tensor(v) for k, v in _centroid_location.items()}

        self._idx_to_key: list = {idx: k for idx, k in enumerate(_centroid_location_dict.keys())}
        self._centroid_location: torch.Tensor = torch.stack(
            list(_centroid_location_dict.values())
        ).to(self._device)

    # maybe ret list[tuple(SIGN, CONFIDENCE: float)] ?
    def classify(self, imgs: List[np.array]) -> List[np.array]:
        res = []
        imgs = [x / 255 for x in imgs]
        transformed_imgs = torch.stack([self._transform(image=img)['image'] for img in imgs])
        transformed_imgs = transformed_imgs.to(self._device, dtype=torch.float32)
        res = self._model(transformed_imgs)
        res1 = self._get_nearest_centroids(res)
        return res1

    def _get_nearest_centroids(self, embs) -> str:
        dist = (embs - self._centroid_location).pow(2).sum(-1).sqrt()
        key = self._idx_to_key[int(torch.argmin(dist))]
        return key


def test():
    device = torch.device('cuda:0')
    c = NonEncoderBasedClassifier(
        path_to_weights='EXCLUDE_ADDI_SIGNSencoder_loss_1e'
        '-05_acc_0.99566epoch_99.encoder',
        path_to_model_config='encoder_config.json',
        path_to_centroid_location='centroid_location.txt',
        device=device
    )

    img = imread_rgb(
        "D:\\d_tsw\\main_diplom\\SignDetectorAndClassifier\\data\\additional_sign\\2.4_1.png"
    )
    sign = c.classify([img])

    return sign


if __name__ == "__main__":
    test()
