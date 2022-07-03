import os

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import logging

from maddrive_adas.train.datasets import ConcatDatasets, PreprocessedDataset, SignsOnlyDataset
from maddrive_adas.train.operations import BboxValidationOp, LetterboxingOp

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

import numpy as np

from maddrive_adas.tools.anchors import YoloAnchorsGenerator


SUBPROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(SUBPROJECT_ROOT, os.pardir, os.pardir))

CONFIG_DPATH = os.path.join(SUBPROJECT_ROOT, "config")


def load_datasets(cfg):
    preproc_ops = [
        LetterboxingOp(target_sz=cfg.model.infer_sz_hw),
        BboxValidationOp(),
    ]

    train_datasets_descs = cfg.datasets.train
    train_datasets = []

    for train_ds_desc in train_datasets_descs:
        if train_ds_desc.type == "signs":
            dataset = SignsOnlyDataset(
                root_dirpath=os.path.join(PROJECT_ROOT, train_ds_desc.path),
                # cache_dirpath=os.path.join(PROJECT_ROOT, "_cache"),
            )
        else:
            raise NotImplementedError(f"Type {train_ds_desc.type} not implemented")

        train_datasets.append(dataset)

    train_dataset = ConcatDatasets(train_datasets)
    train_dataset = PreprocessedDataset(dataset=train_dataset, ops=preproc_ops)

    return train_dataset


@hydra.main(config_path=CONFIG_DPATH, config_name="binary_detector")
def main(cfg: DictConfig):
    train_dataset = load_datasets(cfg)
    target_dims = []

    print(f"Collecting bboxes from {len(train_dataset)} samples")

    for i in tqdm(range(len(train_dataset))):
        _, ann = train_dataset[i]
        bboxes = ann["bboxes"]
        target_dims.extend(bboxes[:, 2:4])

    target_dims = np.array(target_dims, dtype=np.float32)
    target_dims = target_dims[target_dims.sum(axis=1) > 0]

    ANCHORS_COUNT = len(cfg.model.anchor_masks) * len(cfg.model.anchor_masks[0])
    anchors_gen = YoloAnchorsGenerator(ANCHORS_COUNT)

    centroids = anchors_gen.generate(target_dims)
    centroids_str = anchors_gen.centroids_as_string(centroids)

    logger.info(f"Anchors: {centroids_str}")
    print(f"Anchors: {centroids_str}")



if __name__ == "__main__":
    main()
