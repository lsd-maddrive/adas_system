import numpy as np
from typing import List, Tuple

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
    @torch.no_grad()
    def classify(self, imgs: List[np.array]) -> List[Tuple[str, float]]:
        imgs = [x / 255 for x in imgs]
        transformed_imgs = torch.stack([self._transform(image=img)['image'] for img in imgs])
        transformed_imgs = transformed_imgs.to(self._device, dtype=torch.float32)
        model_pred = self._model(transformed_imgs)
        return self._get_nearest_centroids(model_pred)

    def _get_nearest_centroids(self, embs) -> List[str]:
        nearest_sign = []
        for emb in embs:
            dist = (emb - self._centroid_location).pow(2).sum(-1).sqrt()
            dist_as_dict = {idx: v for idx, v in enumerate(dist)}
            sorted_dist_indexes = sorted(dist_as_dict, key=dist_as_dict.get)
            # pick 2 nearest.
            # get conf by (len_to_nearest) / (len_to_nearest + len_to_second_nearest)
            confidence = 2 * dist_as_dict[sorted_dist_indexes[0]] / \
                (dist_as_dict[sorted_dist_indexes[0]] + dist_as_dict[sorted_dist_indexes[1]])
            key = self._idx_to_key[sorted_dist_indexes[0]]
            nearest_sign.append((key, confidence))
        return nearest_sign


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
    img2 = imread_rgb(
        "D:\\d_tsw\\main_diplom\\SignDetectorAndClassifier\\data\\additional_sign\\1.31_1.png"
    )
    img3 = imread_rgb(
        "D:\\d_tsw\\main_diplom\\SignDetectorAndClassifier\\data\\additional_sign\\3.24.100_3.png"
    )

    sign = c.classify([img, img2, img3])

    return sign


if __name__ == "__main__":
    test()
