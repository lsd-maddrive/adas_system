import cv2
import numpy as np
import torch
from ..utils import image as imut
from ..utils.bbox import clip_xywh_bboxes

class BasePreprocessingOp(object):
    @staticmethod
    def deserialize(op_desc, supported_ops):
        TYPES = {op.TYPE: op for op in supported_ops}
        type_ = op_desc["type"]

        for op in supported_ops:
            if type_ != op.TYPE:
                continue

            return TYPES[type_].from_description(**op_desc)

        raise NotImplementedError(f"Method with description {op_desc} not found")

    def transform(self, img: np.ndarray, ann: dict = None, data: dict = {}):
        raise NotImplementedError("Must be implemented")

    def inverse_transform(self, preds: dict, data: dict):
        raise NotImplementedError("Must be implemented")

    def serialize(self):
        raise NotImplementedError("Must be implemented")

    @classmethod
    def from_description(cls, **kwargs):
        return cls(**kwargs)


class Image2TensorOp(BasePreprocessingOp):
    TYPE = "image2tensor"

    def __init__(self, scale_div, **other):
        self.scale_div = scale_div

    def transform(self, img, ann=None, data=None):
        img = torch.from_numpy(img).float().div(self.scale_div).permute(2, 0, 1)

        if ann is not None:
            if "cat_mask" in ann:
                ann["cat_mask"] = (
                    torch.from_numpy(ann["cat_mask"]).long().permute(2, 0, 1)
                )
            if "bboxes" in ann:
                ann["bboxes"] = torch.from_numpy(ann["bboxes"])
            if "labels" in ann:
                ann["labels"] = torch.from_numpy(ann["labels"])
            if "bboxes_labels" in ann:
                ann["bboxes_labels"] = torch.from_numpy(ann["bboxes_labels"])
            if "keypoints" in ann:
                ann["keypoints"] = torch.from_numpy(ann["keypoints"])
            if "instance_masks" in ann:
                ann["instance_masks"] = (
                    torch.from_numpy(ann["instance_masks"]).long().permute(2, 0, 1)
                )
        return img, ann

    def inverse_transform(self, preds: dict, data: dict):
        if 'instance_masks' in preds:
            # Inverse transform mask
            masks = preds['instance_masks']
            masks = masks.numpy()
            if masks.shape[0] > 0:
                masks = masks.astype(np.uint8).transpose(1, 2, 0)
            preds['instance_masks'] = masks
        if "cat_mask" in preds:
            mask = preds['cat_mask']
            mask = mask.numpy().astype(np.uint8)
            preds['cat_mask'] = mask
        if 'keypoints' in preds:
            kps = preds['keypoints']
            preds['keypoints'] = kps.numpy()
        if 'bboxes' in preds:
            bboxes = preds['bboxes']
            preds['bboxes'] = bboxes.numpy()
        if 'classes' in preds:
            classes = preds['classes']
            preds['classes'] = classes.numpy()
        if 'scores' in preds:
            scores = preds['scores']
            preds['scores'] = scores.numpy()
        return preds

    def serialize(self):
        return {"scale_div": self.scale_div}


class LetterboxingOp(BasePreprocessingOp):
    TYPE = "letterboxing"

    def __init__(
        self,
        target_sz,
        fill_value=127,
        mask_fill_value=0,
        image_inter=cv2.INTER_AREA,
        mask_inter=cv2.INTER_NEAREST,
        **other,
    ):
        """Letterboxing operation
        Args:
            target_sz (tuple): Target size (HW - height x width format)
            fill_value (int, optional): Value to fill image after resize. Defaults to 127.
            mask_fill_value (int, optional): Value to fill mask after resize. Defaults to 0.
            image_inter (int, optional): Interpolation method for images
            mask_inter (int, optional): Interpolation method for masks
        """
        self.images_interp = image_inter
        self.mask_interp = mask_inter
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        # TODO - support different size mask
        self.target_sz = np.array(target_sz, dtype=int)

    def transform(self, img: np.ndarray, ann=None, data=None):
        img, img_params = imut.letterbox(
            img,
            inter=self.images_interp,
            target_sz=self.target_sz,
            background=self.fill_value,
        )
        
        if ann is not None:
            # Update
            h, w = img.shape[:2]
            ann["height"] = h
            ann["width"] = w
            if "instance_masks" in ann:
                mask = ann["instance_masks"]
                mask, mask_params = imut.letterbox(
                    mask,
                    inter=self.mask_interp,
                    target_sz=self.target_sz,
                    background=self.mask_fill_value,
                )
                ann["instance_masks"] = mask

            if "cat_mask" in ann:
                mask = ann["cat_mask"]
                mask, mask_params = imut.letterbox(
                    mask,
                    inter=self.mask_interp,
                    target_sz=self.target_sz,
                    background=self.mask_fill_value,
                )
                ann["cat_mask"] = mask

            if "bboxes" in ann:
                bboxes = np.array(ann["bboxes"], dtype=np.float32)
                bboxes = imut.letterbox_boxes(bboxes, img_params)
                bboxes = clip_xywh_bboxes(bboxes, img.shape[:2])
                ann["bboxes"] = bboxes

            if 'keypoints' in ann:
                kps = ann['keypoints']
                kps = imut.letterbox_keypoints(kps, img_params)
                ann['keypoints'] = kps

        if data is not None:
            data["letterboxing"] = img_params
        return img, ann

    def inverse_transform(self, preds: dict, data: dict):
        params = data["letterboxing"]
        if 'instance_masks' in preds:
            masks = preds['instance_masks']
            masks = imut.inverse_letterbox_masks(masks, params, self.mask_interp)
            preds['instance_masks'] = masks
            
        if 'cat_mask' in preds:
            mask = preds['cat_mask']
            mask = imut.inverse_letterbox_masks(mask, params, self.mask_interp)
            preds['cat_mask'] = mask

        if 'bboxes' in preds:
            bboxes = preds['bboxes']
            bboxes = imut.inverse_letterbox_boxes(bboxes, params)
            preds['bboxes'] = bboxes

        if 'bboxes_labels' in preds:
            bboxes = preds['bboxes_labels']
            bboxes = imut.inverse_letterbox_boxes(bboxes, params)
            preds['bboxes_labels'] = bboxes

        if 'keypoints' in preds:
            kps = preds['keypoints']
            kps = imut.inverse_letterbox_keypoints(kps, params)
            preds['keypoints'] = kps

        return preds

    def serialize(self):
        return {
            "target_sz": self.target_sz.tolist(),
            "fill_value": self.fill_value,
            "mask_fill_value": self.mask_fill_value,
            "image_inter": self.images_interp,
            "mask_inter": self.mask_interp,
        }


class BboxValidationOp(BasePreprocessingOp):
    TYPE = "bbox_validation"

    def __init__(self, **other):
        pass

    def _validate_bboxes(self, bboxes, im_shape):
        bboxes[:, 2:4] = bboxes[:, :2] + bboxes[:, 2:4]

        # X, Y side
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, im_shape[1])
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, im_shape[0])
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, :2]

        return bboxes

    def transform(self, img, ann=None, data=None):        
        if ann is not None:
            if "bboxes" in ann and "labels" in ann:
                bboxes = ann["bboxes"]
                labels = ann["labels"]
                if len(bboxes) > 0:
                    # Remove bboxes that are 0 width/height
                    mask = np.prod(bboxes[:, 2:4], axis=1) > 0
                    bboxes = bboxes[mask]
                    ann["labels"] = np.array(labels)[mask]
                    ann["bboxes"] = self._validate_bboxes(bboxes, img.shape[:2])
                else:
                    ann['bboxes'] = np.zeros((0, 4))

            if "bboxes" in ann:
                bboxes = ann["bboxes"]
                if len(bboxes) > 0:
                    # Remove bboxes that are 0 width/height
                    mask = np.prod(bboxes[:, 2:4], axis=1) > 0
                    bboxes = bboxes[mask]
                    
                    ann["bboxes"] = self._validate_bboxes(bboxes, img.shape[:2])
                else:
                    ann['bboxes'] = np.zeros((0, 4))

            if "bboxes_labels" in ann:
                bboxes_labels = ann["bboxes_labels"]
                if len(bboxes_labels) > 0:
                    # Remove bboxes that are 0 width/height
                    mask = np.prod(bboxes_labels[:, 2:4], axis=1) > 0
                    ann["bboxes_labels"] = bboxes_labels = bboxes_labels[mask]

                    if len(bboxes_labels) > 0:
                        ann["bboxes_labels"] = self._validate_bboxes(
                            bboxes_labels, img.shape[:2]
                        )
        return img, ann

    def inverse_transform(self, preds: dict, data: dict):
        return preds

    def serialize(self):
        return {}