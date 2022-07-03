import os
import pickle

import xml.etree.ElementTree as ET
import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import bisect

from maddrive_adas.train.operations import Image2TensorOp

from ..utils.torch_utils import image_2_tensor, array_2_tensor


class BaseImageDataset(object):
    def __init__(self, **config):
        pass

    def __getitem__(self, index):
        """Return pair (image, annotation)
        Args:
            index (int): Index to retrieve pair
        Return:
            img (np.ndarray[h, w, c]): Image from dataset
            ann (Dict[str, Any]): Annotation for image
                bboxes (np.ndarray(n, 4)): Bounding boxes on image
                labels (np.ndarray(n)): Labels for bboxes
                cat_mask (np.ndarray(h, w)): Categorical mask
        """
        raise NotImplementedError("This method must be implemented!")

    def _load_image_rgb(self, fpath: str):
        """Load color image by provided filepath
        Args:
            fpath (str): Filepath im system
        Returns:
            np.ndarray: RGB image
        """

        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class Image2TensorDataset(Dataset):
    def __init__(self, dataset, max_targets=60, scale_div=255):
        self.op = Image2TensorOp(scale_div=scale_div)
        self.dataset = dataset
        self.max_targets = max_targets
        self.scale_div = scale_div

    def __getitem__(self, index):
        img, ann = self.dataset[index]

        bboxes = np.array(ann['bboxes'])
        labels = np.array(ann['labels'])

        # Set from config
        shaped_bboxes = np.zeros([self.max_targets, 5])

        if len(labels) > 0:
            bboxes_labels = np.hstack((bboxes, labels[..., np.newaxis]))
            bboxes_count = min(bboxes_labels.shape[0], self.max_targets)
            shaped_bboxes[:bboxes_count] = bboxes_labels[:bboxes_count]

        img, ann = self.op.transform(img, {'bboxes': shaped_bboxes})
        return img, ann['bboxes']

    def __len__(self):
        return len(self.dataset)


class PreprocessedDataset(BaseImageDataset):
    def __init__(self, dataset: list, ops: list):
        self.dataset = dataset
        self.ops = ops

    def __getitem__(self, index):
        img, ann = self.dataset[index]

        data = {}
        for op in self.ops:
            img, ann = op.transform(img, ann, data)

        return img, ann

    def __len__(self):
        return len(self.dataset)

    def get_ops(self):
        return self.ops


class AugmentedDataset(BaseImageDataset):
    def __init__(self, dataset: list, aug: object):
        self.dataset = dataset
        self.aug = aug

    def __getitem__(self, index):
        img, ann = self.dataset[index]
        img, ann = self.aug.transform(img, ann)

        return img, ann

    def __len__(self):
        return len(self.dataset)

    def get_augmentations(self):
        return self.aug


class ConcatDatasets(BaseImageDataset):
    def __init__(self, datasets: list):
        self.datasets = datasets

        self.common_len = 0
        self.cumulative_sizes = []
        for ds in datasets:
            self.common_len += len(ds)
            self.cumulative_sizes.append(self.common_len)

    def __getitem__(self, index):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        dataset = self.datasets[dataset_idx]

        if dataset_idx > 0:
            index -= self.cumulative_sizes[dataset_idx - 1]

        img, ann = dataset[index]
        return img, ann

    def __len__(self):
        return self.common_len


class DetectionDataset(Dataset):
    def __init__(self, **config):
        self.cfg = config
        self.max_targets = self.cfg.get("max_targets", 10)

        self.labels = self.cfg["labels"]

        self.img_placeholder = None

    def get_raw_img_ann(self, index):
        img = self._get_image(index)
        ann = self._get_annotation(index)

        bboxes = ann["bboxes"]
        labels = ann["labels"]

        target = {
            "boxes": bboxes,
            "labels": labels,
        }

        return img, target

    def _check_bboxes(self, bboxes, im_sz):
        im_h, im_w = im_sz

        bboxes[:, 2:4] = bboxes[:, :2] + bboxes[:, 2:4]

        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, im_w - 1)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, im_h - 1)

        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, :2]

    def get_image_annotation(self, index):
        img = self.get_image(index)
        ann = self.get_annotation(index)

        return img, ann

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img, anns = self.get_image_annotation(index)

        bboxes = anns["bboxes"]
        labels = anns["labels"]

        assert bboxes.shape[0] < self.max_targets, "Overflow of bboxes count: {}".format(
            bboxes.shape[0]
        )

        shaped_bboxes = np.zeros([self.max_targets, 5])

        if len(labels) > 0:
            bboxes_labels = np.hstack((bboxes, labels[..., np.newaxis]))
            bboxes_count = min(bboxes_labels.shape[0], self.max_targets)
            shaped_bboxes[:bboxes_count] = bboxes_labels[:bboxes_count]

        return image_2_tensor(img), array_2_tensor(shaped_bboxes)


class VocDataset(DetectionDataset):
    def __init__(
        self,
        root_dirpath,
        annot_dirname="Annotations",
        img_dirname="Images",
        labels=[],
        cache_dirpath=None,
        mode_train=True,
        dataset=None,
        transform=None,
        dataset_config={},
    ):
        self.cfg = dataset_config
        self.max_targets = self.cfg.get("max_targets", 10)

        self.mode_train = mode_train

        if dataset is None:
            ann_dirpath = os.path.join(root_dirpath, annot_dirname)
            img_dirpath = os.path.join(root_dirpath, img_dirname)

            cache_fpath = None
            if cache_dirpath is not None:
                cache_fname = root_dirpath.replace(".", "")
                cache_fname = cache_fname.replace("/", " ")
                cache_fname = cache_fname.strip()
                cache_fname = cache_fname.replace(" ", "_")
                os.makedirs(cache_dirpath, exist_ok=True)

                cache_fpath = os.path.join(cache_dirpath, cache_fname)

            if cache_fpath is not None and os.path.exists(cache_fpath):
                with open(cache_fpath, "rb") as handle:
                    self.annot_data = pickle.load(handle)
            else:
                self.annot_data = self._parse_voc_annotation(ann_dirpath, img_dirpath, labels)

            if cache_fpath:
                with open(cache_fpath, "wb") as handle:
                    pickle.dump(self.annot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Copy mode
            self.annot_data = dataset.annot_data

        self.transform = transform

        # TODO - remove from here
        self.annot_data, labels_stat = replace_all_labels_2_one(self.annot_data, "sign")

        self.set_labels(labels_stat.keys())

        self.root_dirpath = root_dirpath
        self.img_dirname = img_dirname
        self.data_len = len(self.annot_data)

    def set_labels(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if index >= self.data_len:
            raise ValueError(f"Index {index} overflows data length {self.data_len}")

        ann_data = self.annot_data[index]
        img_fpath = ann_data["filepath"]

        # img_fpath = os.path.join(self.root_dirpath, self.img_dirname, img_fname)
        img = cv2.imread(img_fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = []
        labels = []

        for ann in ann_data["objects"]:
            ann_name = ann["name"]
            xmin = ann["xmin"]
            ymin = ann["ymin"]
            xmax = ann["xmax"]
            ymax = ann["ymax"]

            # COCO format
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            label_id = self.labels.index(ann_name)
            labels.append(label_id)
            bboxes.append(bbox)

        bboxes = np.array(bboxes, dtype=np.float)
        labels = np.array(labels, dtype=np.float)
        if bboxes.shape[0] >= self.max_targets:
            raise ValueError(f"Overflow of bboxes count: {bboxes.shape[0]}")

        if self.transform is not None:
            try:
                transformed = self.transform(image=img, bboxes=bboxes, id=labels)
            except Exception as e:
                print(f"Failed to transform bboxes from img:\n{bboxes} / {labels} ({e})")
                return None, None

            img = transformed["image"]
            bboxes = np.array(transformed["bboxes"])
            labels = np.array(transformed["id"])

        if not self.mode_train:
            target = {
                "boxes": torch.as_tensor(bboxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([index]),
                "area": (bboxes[:, 3]) * (bboxes[:, 2]),
                "iscrowd": torch.zeros((bboxes.shape[0],), dtype=torch.int64),
            }

            return img, target

        # [x, y, w, h, class_id]
        shaped_bboxes = np.zeros([self.max_targets, 5])

        if len(labels) > 0:
            bboxes_labels = np.hstack((bboxes, labels[..., np.newaxis]))
            bboxes_count = min(bboxes_labels.shape[0], self.max_targets)
            shaped_bboxes[:bboxes_count] = bboxes_labels[:bboxes_count]

        # # Preprocessing for training
        # img = torch.from_numpy(img).div(255.0).permute(2, 0, 1)
        # shaped_bboxes = torch.from_numpy(shaped_bboxes)

        return img, shaped_bboxes

    def get_labels_stats(self):
        stats = {}

        for inst in self.annot_data:
            for obj in inst["objects"]:
                if obj["name"] not in stats:
                    stats[obj["name"]] = 0

                stats[obj["name"]] += 1

        return stats

    def _parse_voc_annotation(self, ann_dirpath, img_dirpath, cache_name, labels=[]):
        all_insts = []
        for ann in os.listdir(ann_dirpath):
            img = {"objects": []}
            try:
                ann_fpath = os.path.join(ann_dirpath, ann)
                tree = ET.parse(ann_fpath)
            except Exception as e:
                print(f"Ignore this bad annotation: {ann_fpath} [{e}]")
                continue

            for elem in tree.iter():
                if "filename" in elem.tag:
                    img["filepath"] = os.path.join(img_dirpath, elem.text)

                if "width" in elem.tag:
                    img["width"] = int(elem.text)
                if "height" in elem.tag:
                    img["height"] = int(elem.text)
                if "object" in elem.tag or "part" in elem.tag:
                    obj = {}
                    for attr in list(elem):
                        if "name" in attr.tag:
                            obj["name"] = obj_label = attr.text
                            if labels and obj_label not in labels:
                                print(f"Ignore label: {obj_label}")
                                obj = None
                                break

                        if "bndbox" in attr.tag:
                            for dim in list(attr):
                                if "xmin" in dim.tag:
                                    obj["xmin"] = int(round(float(dim.text)))
                                if "ymin" in dim.tag:
                                    obj["ymin"] = int(round(float(dim.text)))
                                if "xmax" in dim.tag:
                                    obj["xmax"] = int(round(float(dim.text)))
                                if "ymax" in dim.tag:
                                    obj["ymax"] = int(round(float(dim.text)))

                    if obj is not None:
                        img["objects"].append(obj)

            if img is not None:
                all_insts.append(img)

        return all_insts


def replace_all_labels_2_one(instances, new_label):

    labels = {new_label: 0}

    for inst in instances:
        for obj in inst["objects"]:
            obj["name"] = new_label

            labels[new_label] += 1

    return instances, labels
