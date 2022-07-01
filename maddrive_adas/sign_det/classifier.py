from typing import List, Tuple, Union

import json
import subprocess

import torch
import numpy as np
import pytesseract
import cv2

from .base import AbstractSignClassifier, DetectedInstance
from maddrive_adas.utils.transforms import get_minimal_and_augment_transforms
from maddrive_adas.utils.models import get_model_and_img_size

_TARGET_WIDTH = 40
_erode_kernel = np.ones((2, 2), np.uint8)

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
        ignore_tesseract: bool = False,
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
        self._ignore_tesseract = ignore_tesseract
        if not ignore_tesseract:
            try:
                output = subprocess.check_output(
                    'tesseract -v',
                    stderr=subprocess.STDOUT,
                    shell=True,
                ).decode()
                if 'tesseract' not in output:
                    raise subprocess.CalledProcessError
                else:
                    _tesseract_ver_major = int(
                        output.split('\r\n')[0].split()[1].split('.')[0])
                    print(f'Founded tesseract {_tesseract_ver_major}.X.X')

                    if _tesseract_ver_major == 4:
                        self._tesseract_additional_args = '--psm 13 digits'  # TODO: use 6
                    else:
                        self._tesseract_additional_args = '--psm 9'
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
        sign_and_confs_per_image = list(
            map(self._fixup_signs_with_text, imgs, sign_and_confs_per_image)
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

    def _fixup_signs_with_text(
        self,
        img_src: np.ndarray,
        sign_and_confs_for_image: Tuple[str, float],
        ret_debug_img=False,
    ):
        d: List[np.ndarray] = []
        if not self._ignore_tesseract:
            if sign_and_confs_for_image[0] in ['3.24', '3.25']:
                # fixup img
                img = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
                img = crop_img(img, xscale=0.7, yscale=0.4)
                scale_x = _TARGET_WIDTH / img.shape[0]
                img = cv2.resize(img, (int(img.shape[0] * scale_x),
                                       _TARGET_WIDTH), interpolation=cv2.INTER_AREA)
                img = cv2.GaussianBlur(img, (7, 7), 0)
                img = cv2.adaptiveThreshold(
                    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, _erode_kernel, iterations=2)
                img = cv2.erode(img, _erode_kernel, cv2.BORDER_CONSTANT)
                img = cv2.dilate(img, _erode_kernel, cv2.BORDER_CONSTANT, iterations=1)

                tes_out: str = pytesseract.image_to_string(
                    img,
                    config=self._tesseract_additional_args)
                if ret_debug_img:
                    print('appending debug img')
                    d.append(img)

                # if we cannot get output, let's try one more time
                if not tes_out:
                    # oldfix
                    img = crop_img(img_src, xscale=0.7, yscale=0.4)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                    invert = 255 - opening
                    # get tesseract output
                    tes_out: str = pytesseract.image_to_string(
                        invert,
                        config=self._tesseract_additional_args)
                    if ret_debug_img:
                        print('appending debug img')
                        d.append(invert)

                if tes_out:
                    # remove new lines and filter alpha and digits
                    tes_out: str = tes_out.split('\n\n')[0]
                    tes_out = ''.join(filter(lambda w: w.isalpha() or w.isdigit(), tes_out))
                    tes_out = tes_out.lower().replace('j', '1').replace(
                        'l', '1').replace('o', '0').replace('q', '0').replace('t', '1').replace(
                        'c', '0').replace('i', '1').replace('a', '4')

                    sign_and_confs_for_image = (
                        sign_and_confs_for_image[0] + f'.{tes_out}', sign_and_confs_for_image[1]
                    )
                # print(tes_out)
                # cv2.imshow(sign_and_confs_for_image[0] + '_invert', invert)
                # cv2.imshow(sign_and_confs_for_image[0], img)
                # cv2.waitKey(0)
        if ret_debug_img:
            return sign_and_confs_for_image, d
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
