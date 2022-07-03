from typing import List, Tuple, Union
import json

import numpy as np
import torch

from maddrive_adas.utils.models import get_model_and_img_size
from maddrive_adas.utils.transforms import get_minimal_and_augment_transforms
from .base import AbstractSignClassifier, DetectedInstance


REQUIRED_ARCHIVE_KEYS = ['model', 'centroid_location', 'model_config']
SUBC_REQUIRED_KEYS = ['model', 'model_config', 'code_to_sign_dict']


class EncoderBasedClassifier(AbstractSignClassifier):
    """Encoder Bassed Classifier.

    Args:
        BaseSignsClassifier (AbstractClassifier): Abstract Classifier.
    """

    def __init__(
        self,
        config_path: str,
        path_to_centroid_location: dict = None,
        path_to_subclassifier_3_24_and_3_25_config: str = '',
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
        # get device
        self._device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self._use_subclassifier: bool = False
        # initialize subclassifier if it was passed
        if path_to_subclassifier_3_24_and_3_25_config:
            print('[+] Loading subclassifier')
            self._use_subclassifier: bool = True
            subclassifier_dict: dict = torch.load(
                path_to_subclassifier_3_24_and_3_25_config,
                map_location=torch.device('cpu')
            )
            assert(all([key in subclassifier_dict.keys() for key in SUBC_REQUIRED_KEYS])
                   ), f'Verify subclassifier archive keys. It should contain {SUBC_REQUIRED_KEYS}'
            self._subc, self._subc_img_size = get_model_and_img_size(
                config_data=subclassifier_dict['model_config']
            )
            self._subc_transform, _ = get_minimal_and_augment_transforms(
                self._subc_img_size, interpolation=3)   # 3 ~ interpolation=cv2.INTER_AREA

            self._subc.load_state_dict(subclassifier_dict['model'])
            self._subc = self._subc.to(self._device)
            self._subc.eval()
            # warmup
            self._subc(torch.rand(
                (1, 3, self._subc_img_size, self._subc_img_size)).to(self._device)
            )
            self._subc_code_to_sign_dict: dict = subclassifier_dict['code_to_sign_dict']
        else:
            print('[!] You are running WO 3.24, 3.25 subclassifier')

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

    # noqa
    @torch.no_grad()
    def classify_batch(
        self,
        detected_instances: List[DetectedInstance],
        **kwargs
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
                # print('[!] Passed for classification data is not isntance of DetectedInstacnce')
                # print("[!] It's np.ndarray. Trying to append it as raw image for classification")
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
        sign_and_confs_per_image = list(
            map(self._fixup_speed_signs, imgs, sign_and_confs_per_image)
        )

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

    def _fixup_speed_signs(
        self,
        img_src: np.ndarray,
        sign_and_confs_for_image: Tuple[str, float],
    ) -> Tuple[str, float]:
        """Fix 3.24 and 3.25 return signs.

        Args:
            img_src (np.ndarray): passed img.
            sign_and_confs_for_image (Tuple[str, float]): Predicted sign and conf.

        Returns:
            Tuple[str, float]: possible fixed value.
        """
        if self._use_subclassifier:
            if sign_and_confs_for_image[0] in ['3.24', '3.25']:
                # print(sum(sum(img_src)))
                model_input = torch.stack(
                    [self._subc_transform(image=img_src)['image'] / 255]
                ).to(self._device)
                # print(self._subc_transform)
                # print(torch.sum(model_input))
                pred = self._subc(model_input)
                pred_softmaxed = torch.softmax(pred, dim=1)[0]
                # print(pred_softmaxed)
                argmax = torch.argmax(pred_softmaxed).item()
                # print(argmax)
                predicted_sign = self._subc_code_to_sign_dict[argmax]
                conf = pred_softmaxed[argmax]
                return (predicted_sign, conf.item())

        return sign_and_confs_for_image

    def classify(
        self,
        instance: Union[DetectedInstance, np.ndarray],
        **kwargs
    ) -> Tuple[DetectedInstance, List[Tuple[str, float]]]:
        """Classify a single DetectedInstance.

        Args:
            instance (DetectedInstance): DetectedInstance image description.

        Returns:
            Tuple[str, float]: (sign, confidence)
        """
        return self.classify_batch([instance])[0]


def crop_img(img, xscale=1.0, yscale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * xscale, img.shape[0] * yscale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped
