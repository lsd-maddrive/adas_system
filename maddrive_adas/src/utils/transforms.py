import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import PadIfNeeded
from albumentations.augmentations.geometric.resize import LongestMaxSize

def get_minimal_and_augment_transforms(img_size):
    minimal_transform = A.Compose(
            [
            LongestMaxSize(img_size),
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
                               value=(0,0,0),
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
            LongestMaxSize(img_size),
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
