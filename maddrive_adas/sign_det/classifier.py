from typing import List, Tuple, Union

import json
import subprocess

import torch
import numpy as np

from .base import AbstractSignClassifier, DetectedInstance
from maddrive_adas.utils.transforms import get_minimal_and_augment_transforms
from maddrive_adas.utils.models import get_model_and_img_size


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
        try:
            output = subprocess.check_output(
                'tesseract -v',
                stderr=subprocess.STDOUT
            ).decode()
            if 'tesseract' not in output:
                raise subprocess.CalledProcessError
        except subprocess.CalledProcessError:
            print('Unable to call tessecact. Install and add tesseract to PATH variable.')
            print('Link: https://tesseract-ocr.github.io/tessdoc/Downloads.html')
            raise subprocess.CalledProcessError

        self._device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        model_dict: dict = torch.load(config_path, map_location=torch.device('cpu'))
        assert(all([key in model_dict.keys() for key in REQUIRED_ARCHIVE_KEYS])
               ), f'Verify model archive keys. It should contain {REQUIRED_ARCHIVE_KEYS}'

        self._model, self._img_size = get_model_and_img_size(config_data=model_dict['model_config'])
        self._model.load_state_dict(model_dict['model'])
        self._model = self._model.to(self._device)
        self._model.eval()

        self._transform, _ = get_minimal_and_augment_transforms(self._img_size)

        _centroid_location_dict: dict = path_to_centroid_location \
            if path_to_centroid_location else model_dict['centroid_location']

        # TODO: fix ME. multiple json translations
        while isinstance(_centroid_location_dict, str):
            _centroid_location_dict = json.loads(_centroid_location_dict)

        self._idx_to_key: list = {idx: k for idx, k in enumerate(_centroid_location_dict.keys())}
        self._centroid_location: torch.Tensor = torch.stack(
            [torch.Tensor(v) for v in _centroid_location_dict.values()]
        ).to(self._device)

    @torch.no_grad()
    def classify_batch(
        self,
        detected_instances: List[DetectedInstance]
    ) -> List[Tuple[DetectedInstance, List[Tuple[str, float]]]]:
        """Classify image batch.

        Args:
            instances (List[DetectedInstance]): List of DetectedInstance image
            descriptions.

        Returns:
            List[Tuple[str, float]]: List of tuple(sign, confidence)
        """
        if not detected_instances:
            return []

        # 2. crop img and make array from it
        # TODO: make generator from DetectedInstance aka yield
        imgs: List[np.ndarray] = []
        for detected_instance in detected_instances:
            if isinstance(detected_instance, DetectedInstance):
                for idx in range(0, detected_instance.get_roi_count()):
                    imgs.append(detected_instance.get_cropped_img(idx))
            elif isinstance(detected_instance, np.ndarray):
                print('[!] Passed for classification data is not isntance of DetectedInstacnce')
                print("[!] It's np.ndarray. Trying to append it as raw image for classification")
                imgs.append(detected_instance)
            else:
                raise ValueError('Wrong instance type')

        if not imgs:
            return [(x, []) for x in detected_instances]

        # 3. pass it to model
        transformed_imgs = torch.stack([self._transform(image=img)['image'] / 255 for img in imgs])
        transformed_imgs = transformed_imgs.to(self._device, dtype=torch.float32)
        model_pred = self._model(transformed_imgs)

        # 4. get nearest centroid for all img in imgs
        sign_and_confs_per_image: List[str, float] = self._get_sign_and_confidence(model_pred)

        # 4.5. Fix 3.24, 3.25. Get text from image.
        self._fixup_signs_with_text(sign_and_confs_per_image)

        # 5. rearrange to detections per DetectedInstance
        res_per_detected_instance: List[DetectedInstance, List[Tuple[str, float]]] = []
        accum: int = 0
        for d in detected_instances:
            if isinstance(d, DetectedInstance):
                roi_count: int = d.get_roi_count()
                res_per_detected_instance.append(
                    (
                        d,
                        [x for x in sign_and_confs_per_image[accum: accum + roi_count]]
                    )
                )
                accum += roi_count
            elif isinstance(d, np.ndarray):
                _detected_instance = DetectedInstance(d)
                conf = sign_and_confs_per_image[accum: accum + 1]
                _detected_instance.add_rel_roi([0., 0., 1., 1.], conf)
                res_per_detected_instance.append(
                    (
                        _detected_instance,
                        conf
                    )
                )
                accum += 1
        return res_per_detected_instance

    def _get_sign_and_confidence(self, embs) -> List[Union[str, float]]:
        nearest_sign = []
        for emb in embs:
            dist = (emb - self._centroid_location).pow(2).sum(-1).sqrt()
            dist_dict = {idx: v for idx, v in enumerate(dist)}
            sorted_dist_idxies = sorted(dist_dict, key=dist_dict.get)
            # TODO: fix confidence
            # pick 2 nearest.
            # get conf by (len_to_nearest) / (len_to_nearest + len_to_second_nearest)
            confidence = (2 * dist_dict[sorted_dist_idxies[0]] / (
                dist_dict[sorted_dist_idxies[0]] + dist_dict[sorted_dist_idxies[1]]))
            key = self._idx_to_key[sorted_dist_idxies[0]]
            nearest_sign.append((key, float(confidence)))
        return nearest_sign

    @staticmethod
    def _fixup_signs_with_text(sign_and_confs_per_image: List[Union[str, float]]):
        for v in sign_and_confs_per_image:
            if v[0] in ['3.24', '3.25']:
                # TODO: implement tesseract
                pass

    def classify(
        self,
        instance: Union[DetectedInstance, np.ndarray],
    ) -> Tuple[DetectedInstance, List[Tuple[str, float]]]:
        """Classify a single DetectedInstance.

        Args:
            instance (DetectedInstance): DetectedInstance image description.

        Returns:
            Tuple[str, float]: (sign, confidence)
        """
        return self.classify_batch([instance])[0]
