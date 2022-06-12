from pathlib import Path
import json

import torch
import numpy as np

from src.utils.fs import imread_rgb
from src.utils.transforms import get_minimal_and_augment_transforms
from src.utils.models import get_model_and_img_size

from base import AbstractSignClassifier

REQUIRED_ARCHIVE_KEYS = ['model', 'centroid_location', 'model_config']


class EncoderBasedClassifier(AbstractSignClassifier):
    """Encoder Bassed Classifier.

    Args:
        BaseSignsClassifier (AbstractClassifier): Abstract Classifier.
    """

    def __init__(
        self,
        config_path: str,
        path_to_centroid_location: dict = None,
        device: torch.device = None,
    ):
        """EncoderBasedClassifier Constructor.

        Args:
            path_to_model_archive (str): Path to model archive, which contains:

                - model config json - config;
                - model weights - model;
                - centroid locations - centrod;
            path_to_centroid_location (dict, optional): Pass dict centroid location for overwriting
            centroids from model archive. Defaults to ''.
        """
        self._device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        model_dict: dict = torch.load(config_path, map_location=self._device)
        assert(all([key in model_dict.keys() for key in REQUIRED_ARCHIVE_KEYS])
               ), f'Verify model archive keys. It should contain {REQUIRED_ARCHIVE_KEYS}'

        self._model, self._img_size = get_model_and_img_size(config_data=model_dict['model_config'])
        self._model.load_state_dict(model_dict['model'])
        self._transform, _ = get_minimal_and_augment_transforms(self._img_size)

        _centroid_location_dict: dict = path_to_centroid_location \
            if path_to_centroid_location else json.loads(
                json.loads(model_dict['centroid_location'])
            )
        self._idx_to_key: list = {idx: k for idx, k in enumerate(_centroid_location_dict.keys())}
        self._centroid_location: torch.Tensor = torch.stack(
            [torch.Tensor(v) for v in _centroid_location_dict.values()]
        ).to(self._device)

    @torch.no_grad()
    def classify_batch(
        self,
        imgs: list[np.array],
        relative_sign_pos: list[list[float]]
    ) -> list[tuple[str, float]]:

        # 1. to float
        imgs_float = [x / 255 for x in imgs]
        # 2. crop img and make array from it
        imgs: list[np.array] = []
        for idx, img in enumerate(imgs_float):
            for sign_pos in relative_sign_pos[idx]:
                w, h, *_ = img.shape
                imgs.append(
                    img[
                        int(sign_pos[0] * w): int(sign_pos[1] * w),
                        int(sign_pos[0] * h): int(sign_pos[1] * h),
                    ],
                )
        # 3. pass it to model
        transformed_imgs = torch.stack([self._transform(image=img)['image'] for img in imgs])
        transformed_imgs = transformed_imgs.to(self._device, dtype=torch.float32)
        model_pred = self._model(transformed_imgs)

        # 4. get nearest
        return self._get_nearest_centroids(model_pred)

    def _get_nearest_centroids(self, embs) -> list[str]:
        nearest_sign = []
        for emb in embs:
            dist = (emb - self._centroid_location).pow(2).sum(-1).sqrt()
            dist_dict = {idx: v for idx, v in enumerate(dist)}
            sorted_dist_idxies = sorted(dist_dict, key=dist_dict.get)
            # TODO: fix confidence
            # pick 2 nearest.
            # get conf by (len_to_nearest) / (len_to_nearest + len_to_second_nearest)
            # confidence = (2 * dist_dict[sorted_dist_idxies[0]] /
            #               (dist_dict[sorted_dist_idxies[0]] + dist_dict[sorted_dist_idxies[1]]))
            confidence = 1
            key = self._idx_to_key[sorted_dist_idxies[0]]
            nearest_sign.append((key, confidence))
        return nearest_sign


def test():
    PROJECT_ROOT = Path('.')
    DATA_DIR = PROJECT_ROOT / 'SignDetectorAndClassifier' / 'data'
    MODEL_ARCHIVE = PROJECT_ROOT / 'maddrive_adas' / 'sign_det' / 'encoder_cl_config'

    c: AbstractSignClassifier = EncoderBasedClassifier(config_path=str(MODEL_ARCHIVE))

    img1 = imread_rgb(DATA_DIR / 'additional_sign' / '2.4_1.png')
    img2 = imread_rgb(DATA_DIR / 'additional_sign' / '1.31_1.png')
    img3 = imread_rgb(DATA_DIR / 'additional_sign' / '3.24.100_3.png')

    sign = c.classify_batch(
        [img1, img2, img3],
        [[[0., 1., 0., 1.]] for _ in range(3)]
    )

    return sign


if __name__ == "__main__":
    test()
