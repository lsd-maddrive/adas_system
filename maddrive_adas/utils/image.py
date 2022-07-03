from typing import Tuple, Union, Any
import math

import numpy as np
import cv2


class LetterboxResizeParams(object):
    def __init__(self, padding, scale, source_sz, embbeded_sz):
        super().__init__()
        self.padding = padding
        self.scale = scale
        self.source_sz = source_sz
        self.embbeded_sz = embbeded_sz


def letterbox(
    img: np.ndarray,
    target_sz: Tuple[int, int],
    background: Union[int, Tuple[int, int, int]],
    inter: int = cv2.INTER_CUBIC,
):
    """Image resize method letterboxing (embedding image into target size)
    Args:
        img (np.ndarray): Image to letterbox
        inter (int): Interpolation method (based on cv2.INTER_*)
        target_sz (Tuple[int, int]): Target size of image (HW format)
        background (int): Value to fill zone not covered with image
    Returns:
        Tuple[np.ndarray, LetterboxResizeParams]: Letterboxed image with resize parameters for recovery
    """

    # assert img.dtype == np.uint8, "Now only uint8 images are supported"

    if len(img.shape) > 2:
        channels = img.shape[2]
    else:
        channels = 1

    source_size = np.array(img.shape[:2])

    new_sz = source_size * min(target_sz / source_size)
    new_sz = new_sz.astype(int)

    # scale [y, x]
    scale = new_sz / source_size
    # Must be width, height
    img = cv2.resize(img, dsize=tuple(map(int, new_sz[::-1])), interpolation=inter)

    if len(img.shape) < 3:
        img = img[..., None]

    padding = target_sz - new_sz
    pad = padding.astype(int) // 2

    new_img = np.full(
        shape=(target_sz[0], target_sz[1], channels),
        dtype=img.dtype,
        fill_value=background,
    )
    new_img[pad[0] : pad[0] + img.shape[0], pad[1] : pad[1] + img.shape[1], :] = img

    params = LetterboxResizeParams(pad, scale, source_size, new_sz)

    return new_img, params


def letterbox_boxes(boxes, params: LetterboxResizeParams):
    """Transform boxes with parameters generated from image letterboxing
    Args:
        boxes (np.ndarray): Boxes in COCO format (XYWH format)
        params (LetterboxResizeParams): Parameters of image letterboxing
    Returns:
        np.ndarray: Transformed boxes
    """
    if boxes.shape[0] > 0:
        boxes[:, [1, 3]] *= params.scale[0]
        boxes[:, [0, 2]] *= params.scale[1]
        boxes[:, [1, 0]] += params.padding

    return boxes

def letterbox_keypoints(kps, params: LetterboxResizeParams):
    """Transform keypoints with parameters generated from image letterboxing
    Args:
        kps (np.ndarray): Keypoints in COCO format, shape=[n_humans, n_keypoints, (x, y, visible, score)]
        params (LetterboxResizeParams): Parameters of image letterboxing
    Returns:
        np.ndarray: Transformed keypoints
    """
    if kps.shape[0] > 0:
        kps[..., [1, 0]] = (kps[..., [1, 0]] * params.scale) + params.padding

    return kps

def inverse_letterbox_boxes(boxes, params: LetterboxResizeParams):
    """Inverse transform boxes with parameters generated from image letterboxing
    Args:
        boxes (np.ndarray): Boxes in COCO format (XYWH format)
        params (LetterboxResizeParams): Parameters of image letterboxing
    Returns:
        np.ndarray: Transformed boxes
    """
    if boxes.shape[0] > 0:
        boxes[:, [1, 0]] -= params.padding
        boxes[:, [1, 3]] /= params.scale[0]
        boxes[:, [0, 2]] /= params.scale[1]

    return boxes


def inverse_letterbox_masks(masks, params: LetterboxResizeParams, iterpolation: int):
    """Inverse transform masks with parameters generated from image letterboxing
    Args:
        masks (np.ndarray): Masks to transform
        params (LetterboxResizeParams): Parameters of image letterboxing
        iterpolation (int): Interpolation method
    Returns:
        np.ndarray: Transformed bboxes
    """
    if masks.shape[0] > 0:
        pad = params.padding
        emb_sz = params.embbeded_sz
        masks = masks[pad[0] : pad[0] + emb_sz[0], pad[1] : pad[1] + emb_sz[1]]
        masks = cv2.resize(
            masks,
            dsize=None,
            fx=1 / params.scale[1],
            fy=1 / params.scale[0],
            interpolation=iterpolation,
        )
        if len(masks.shape) < 3:
            masks = masks[..., None]
    return masks

def inverse_letterbox_keypoints(kps, params: LetterboxResizeParams):
    """Inverse transform keypoints with parameters generated from image letterboxing
    Args:
        kps (np.ndarray): Keypoints in COCO format, shape=[n_humans, n_keypoints, (x, y, visible, score)]
        params (LetterboxResizeParams): Parameters of image letterboxing
    Returns:
        np.ndarray: Transformed keypoints
    """
    if kps.shape[0] > 0:
        kps[..., [1, 0]] = (kps[..., [1, 0]] - params.padding) / params.scale
    return kps

def rotate(
    image: np.ndarray,
    angle: float,
    background: Union[int, Tuple[int, int, int]],
    inter: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Rotate image in source shape
    Args:
        image (np.ndarray): Source image to rotate
        angle (float): Degree angle, positive -> ccw direction
        background (Union[int, Tuple[int, int, int]]): Fill color
        inter (int): Interpolation method
    Returns:
        np.ndarray: Rotated image
    """
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width
    )
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height
    )

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2

    if isinstance(background, (int, float)):
        background = tuple([background] * 3)

    return cv2.warpAffine(
        image,
        rot_mat,
        (int(round(height)), int(round(width))),
        borderValue=background,
        flags=inter,
    )


def imread_gif(fpath):
    import imageio

    gif = imageio.mimread(fpath)
    img = np.array(gif[0], dtype=np.uint8)
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img
