from typing import List, Tuple

import cv2
import torch
import numpy as np
import pandas as pd
from albumentations import CoarseDropout, Compose

from maddrive_adas.utils.augmentations import (
    Albumentations,
    augment_hsv,
    letterbox,
    random_perspective,
)
from maddrive_adas.utils.general import (
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from maddrive_adas.utils.transforms import ConvertCenterXYWH2CV2Rectangle, UnmakeRel
from maddrive_adas.utils.general import LOGGER

import random


def resize_triangle(triangle_pts: List[int], k_limit=0.5, randomize_k=True) -> List[int]:
    """Resize right trianble.

    Args:
        triangle_pts (list[int]): Array of triangle points.
        k_limit (float, optional): Triangle border scale limit. If randomize k, apply scale [0; k), else apply k. Defaults to 0.5.
        randomize_k (bool, optional): Apply random k scale from range [0; k]. Defaults to True.

    Returns:
        list[int]: List of triangle points.
    """
    xs, ys = [x[0] for x in triangle_pts], [x[1] for x in triangle_pts]

    base_x, base_y = None, None
    offset_x, offset_y = None, None

    for x in xs:
        if xs.count(x) == 2:
            base_x = x
        else:
            offset_x = x
    offset_x -= base_x

    for x in ys:
        if ys.count(x) == 2:
            base_y = x
        else:
            offset_y = x
    offset_y -= base_y

    if base_x is None and base_y is None:
        raise ValueError('Unable to get base triangle point. Is it right triangle?')

    x_scale = k_limit * random.random() if randomize_k else k_limit
    y_scale = k_limit * random.random() if randomize_k else k_limit
    return [(base_x, base_y), (int(base_x + offset_x * x_scale), base_y), (base_x, int(base_y + offset_y * y_scale))]


def get_all_rectangle_points(two_rectangle_points: List[int]) -> List[Tuple[int]]:
    """Return TL, TR, BR, BL points."""
    p1 = (two_rectangle_points[0], two_rectangle_points[1])
    p2 = (two_rectangle_points[0], two_rectangle_points[3])
    p3 = (two_rectangle_points[2], two_rectangle_points[3])
    p4 = (two_rectangle_points[2], two_rectangle_points[1])
    return [p1, p2, p3, p4]


def hide_corner_aug(
        img: np.ndarray,
        sign_coordinates_xywh_rel: List[List[int]],
        p=0.9, k_limit=1., randomize_k=True) -> np.ndarray:
    """Hide sign corner.TODO:

    Args:
        img (np.ndarray): _description_
        sign_coordinates_xywh_rel (list[list[int]]): _description_
        p (float, optional): _description_. Defaults to 0.9.
        k_limit (_type_, optional): _description_. Defaults to 1..
        randomize_k (bool, optional): _description_. Defaults to True.

    Returns:
        np.ndarray: Image with hidden corner.
    """
    h, w, _ = img.shape
    for coords in sign_coordinates_xywh_rel:
        if random.random() > p:
            continue
        abs_coords = UnmakeRel(coords, w, h)
        rectangle_coords = ConvertCenterXYWH2CV2Rectangle(abs_coords)
        triangle_coordinates: list[int] = random.sample(
            get_all_rectangle_points(rectangle_coords), 3)
        triangle_coordinates = resize_triangle(
            triangle_coordinates, k_limit=k_limit, randomize_k=randomize_k)
        # print(triangle_coordinates)
        pts = np.array(triangle_coordinates, np.int32)
        # color_const = random.randrange(80, 256)
        color = [random.randrange(80, 256), random.randrange(
            80, 256), random.randrange(80, 256)]  # [color_const] * 3
        cv2.drawContours(img, [pts], 0, color, -1)
    return img


def put_coarse_dropouts(img: np.ndarray, rectangle_coords: List[int], color: List[int], p=0.5):
    sub_img = img[rectangle_coords[1]:rectangle_coords[3],
                  rectangle_coords[0]:rectangle_coords[2], :]
    h, w, d = sub_img.shape
    aug_compose = Compose([
        CoarseDropout(
            max_height=h // 5,
            max_width=w // 5,
            min_holes=1,
            max_holes=10,
            fill_value=color,
            p=p), ]
    )
    coarsed_sub_img = aug_compose(image=sub_img)['image']
    img[rectangle_coords[1]:rectangle_coords[3], rectangle_coords[0]
        :rectangle_coords[2], :] = coarsed_sub_img

    return img


def put_shieeet_on_img_like_winter(img: np.ndarray, sign_coordinates_xywh_rel: List[List[int]]):
    h, w, d = img.shape
    for coords in sign_coordinates_xywh_rel:
        abs_coords = UnmakeRel(coords, w, h)
        rectangle_coords = ConvertCenterXYWH2CV2Rectangle(abs_coords)
        # print(rectangle_coords)
        color_const = random.randrange(80, 256)
        color = [color_const] * 3
        img = put_coarse_dropouts(img, rectangle_coords, color=color, p=1)

    return img


class WinterizedYoloDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        set_label: str,
        hyp_arg: dict,
        img_size=640,
        augment=False,
        hide_corner_chance=0.5
    ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp_arg
        self.df = df[df["set"] == set_label]
        self._hide_corner_chance = hide_corner_chance
        self.albumentations = Albumentations() if augment else None

    def load_image(self, instance):
        path, (w0, h0) = instance["filepath"], instance["size"]
        img = cv2.imread(str(path))
        assert img is not None, f"Image Not Found {path}"
        img = hide_corner_aug(img, instance['coords'], p=self._hide_corner_chance, k_limit=1)
        try:
            img = put_shieeet_on_img_like_winter(img, instance['coords'])
        except ValueError as e:
            msg = f'Cannot winterize {path}. Skipping'
            LOGGER.warning(msg)
        r = self.img_size / max(h0, w0)  # ratio

        if r != 1:  # if sizes are not equal
            img = cv2.resize(
                img,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA
                if r < 1 and not self.augment
                else cv2.INTER_LINEAR,
            )
        return img, (h0, w0), img.shape[:2]

    def __getitem__(self, index):
        # locate img info from DataFrame
        instance = self.df.iloc[index]

        # get Img, src height, width and resized height, width
        try:
            img, (h0, w0), (h, w) = self.load_image(instance)
        except ValueError as e:
            raise ValueError(f'VE for {e}: index is {index}. {instance}')

        shape = self.img_size

        # make img square
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

        # store core shape info
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # add class to labels. We have 1 class, so just add zeros into first column
        labels = np.array(instance["coords"])
        labels = np.c_[np.zeros(labels.shape[0]), labels]

        # fix labels location caused by letterbox
        labels[:, 1:] = xywhn2xyxy(
            labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
        )

        if self.augment:
            img, labels = random_perspective(
                img,
                labels,
                degrees=self.hyp["degrees"],
                translate=self.hyp["translate"],
                scale=self.hyp["scale"],
                shear=self.hyp["shear"],
                perspective=self.hyp["perspective"],
            )

        labels[:, 1:5] = xyxy2xywhn(
            labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=False, eps=1e-3
        )

        # YOLO augmentation technique (!copy-paste!)
        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(
                img,
                hgain=self.hyp["hsv_h"],
                sgain=self.hyp["hsv_s"],
                vgain=self.hyp["hsv_v"],
            )

        nl = len(labels)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, instance["filepath"], shapes

    def __len__(self):
        return len(self.df.index)

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
