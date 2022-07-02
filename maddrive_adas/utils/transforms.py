
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import PadIfNeeded
from albumentations.augmentations.geometric.resize import LongestMaxSize


def get_minimal_and_augment_transforms(img_size, interpolation=cv2.INTER_LINEAR):
    minimal_transform = A.Compose(
        [
            LongestMaxSize(img_size, interpolation=interpolation),
            PadIfNeeded(
                img_size,
                img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            ToTensorV2(),
        ]
    )

    augment_transform = A.Compose(
        [
            A.Blur(blur_limit=2),
            A.CLAHE(p=0.5),
            A.Perspective(scale=(0.01, 0.1), p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05,
                               scale_limit=0.05,
                               interpolation=cv2.INTER_LANCZOS4,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=(0, 0, 0),
                               rotate_limit=6, p=0.5),
            A.RandomGamma(
                gamma_limit=(50, 130),
                p=1
            ),
            A.ImageCompression(quality_lower=80, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.3,
                                       brightness_by_max=False,
                                       p=0.5),
            A.CoarseDropout(max_height=3,
                            max_width=3,
                            min_holes=1,
                            max_holes=3,
                            p=0.5),
            LongestMaxSize(img_size, interpolation=interpolation),
            PadIfNeeded(
                img_size,
                img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            ToTensorV2(),
        ]
    )

    return minimal_transform, augment_transform


def UnmakeRel(coords, w, h):
    return list(map(int, [coords[0] * w, coords[1] * h, coords[2] * w, coords[3] * h]))


def MakeRel(coords, w, h):
    return list(
        map(float, [coords[0] / w, coords[1] / h, coords[2] / w, coords[3] / h])
    )


def ConvertAbsTLWH2CV2Rectangle(coords):
    return list(
        map(int, [coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]])
    )


def ConvertCenterXYWH2CV2Rectangle(coords):
    return list(
        map(
            int,
            [
                coords[0] - coords[2] / 2,
                coords[1] - coords[3] / 2,
                coords[0] + coords[2] / 2,
                coords[1] + coords[3] / 2,
            ],
        )
    )


def ConvertCV2Rectangle2CenterXYWH(coords):
    return list(
        map(
            int,
            [
                (coords[2] + coords[0]) / 2,
                (coords[3] + coords[1]) / 2,
                coords[2] - coords[0],
                coords[3] - coords[1],
            ],
        )
    )


def printProgressEnum(index, length, label=None):
    print(
        "\r{}Progress: {}/{} ({:.2f}%)".format(
            label if label != None else "",
            index + 1,
            length,
            100 * (index + 1) / length,
        ),
        flush=True,
        end="",
    )


def showTensorPicture(tensor_image, label=None):
    # img = tensor_image.permute(1, 2, 0)
    img = cv2.cvtColor(tensor_image.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    if label:
        plt.title(label)
    plt.show()


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
