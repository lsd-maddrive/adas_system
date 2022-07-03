import logging
import os

import hydra
import numpy as np
import torch
from catalyst import dl, metrics
from catalyst import utils as cata_ut


from clearml import Task
from omegaconf import DictConfig, OmegaConf
import hashlib
import json

from typing import Union

import albumentations as albu
import cv2
import numpy as np

from torch import nn
from torch.utils.data import DataLoader

from maddrive_adas.train.datasets import (
    Image2TensorDataset,
    PreprocessedDataset,
    SignsOnlyDataset,
    ConcatDatasets,
    AugmentedDataset,
)
from maddrive_adas.train import callbacks as cb
from maddrive_adas.train.operations import BboxValidationOp, LetterboxingOp, Image2TensorOp
from maddrive_adas.train.utils import construct_model

SUBPROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(SUBPROJECT_ROOT, os.pardir, os.pardir))

CONFIG_DPATH = os.path.join(SUBPROJECT_ROOT, "config")

logger = logging.getLogger(__name__)


def get_loaders(cfg: DictConfig, preproc_ops: list):
    num_workers = get_n_workers(cfg.num_workers)
    if num_workers > cfg.batch_size:
        logger.info(f"Redefine `num_workers` from {num_workers} to {cfg.batch_size}")
        num_workers = cfg.batch_size

    # Train part

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

    train_augmentations = AlbuAugmentation()
    train_dataset = AugmentedDataset(dataset=train_dataset, aug=train_augmentations)

    train_dataset = Image2TensorDataset(train_dataset, max_targets=10)

    train_cfg = {
        "dataset": train_dataset,
        "batch_size": cfg.batch_size,
        "num_workers": num_workers,
        "shuffle": True,
    }
    train_loader = DataLoader(**train_cfg)

    # Valid part

    valid_datasets_descs = cfg.datasets.valid
    valid_datasets = []

    for valid_ds_desc in valid_datasets_descs:
        if valid_ds_desc.type == "signs":
            dataset = SignsOnlyDataset(
                root_dirpath=os.path.join(PROJECT_ROOT, valid_ds_desc.path),
                # cache_dirpath=os.path.join(PROJECT_ROOT, "_cache"),
            )
        else:
            raise NotImplementedError(f"Type {valid_ds_desc.type} not implemented")

        valid_datasets.append(dataset)

    valid_dataset = ConcatDatasets(valid_datasets)
    valid_dataset = PreprocessedDataset(dataset=valid_dataset, ops=preproc_ops)
    valid_dataset = Image2TensorDataset(valid_dataset, max_targets=10)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    logger.info(f"Train dataset length: {len(train_dataset)}")
    logger.info(f"Valid dataset length: {len(valid_dataset)}")

    return {"train": train_loader, "valid": valid_loader}


def get_n_workers(num_workers: int) -> int:
    import multiprocessing as mp

    max_cpu_count = mp.cpu_count()
    if num_workers < 0:
        num_workers = max_cpu_count
        logger.info(f"Parameter `num_workers` is set to {num_workers}")

    num_workers = min(max_cpu_count, num_workers)

    return num_workers


def setup_train_project(project_root: str, seed: int):
    CHECKPOINTS_DPATH = os.path.join(project_root, "torch_checkpoints")

    # Setup checkpoints upload directory
    torch.hub.set_dir(CHECKPOINTS_DPATH)

    cata_ut.set_global_seed(seed)
    cata_ut.torch.prepare_cudnn(deterministic=True, benchmark=True)


def get_hydra_logs_dpath() -> str:
    return os.path.join(os.getcwd(), "logs")


def dict_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg)
    hash_str = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()
    return hash_str


def init_clearml(cfg: DictConfig):
    if not cfg.enable_clearml:
        return None

    cfg_hash = dict_hash(OmegaConf.to_container(cfg, resolve=True))

    tags = []

    if "logger_tags" in cfg:
        for tag in cfg.logger_tags:
            tags.append(tag)

    experiment_name = f"{cfg_hash[:8]}"

    clearml_task = Task.init(
        project_name=cfg.project_name,
        task_name=experiment_name,
        reuse_last_task_id=False,
    )
    clearml_task.add_tags(tags)

    description = ""
    description += f"Config hash: {cfg_hash}\n"

    clearml_task.set_comment(description)

    return clearml_task


@hydra.main(config_path=CONFIG_DPATH, config_name="binary_detector")
def main(cfg: DictConfig):  # noqa: C901
    for env_name, value in cfg.envs.items():
        os.environ[env_name] = value

    setup_train_project(PROJECT_ROOT, seed=cfg.seed)

    # Custom OmegaConf resolver
    OmegaConf.register_new_resolver("len", lambda data: len(data))

    clearml_task = init_clearml(cfg)

    # LOG_ON_BATCH = False
    LOGS_DPATH = get_hydra_logs_dpath()
    checkpoint_dname = os.path.join("train_checkpoints", "signs_detector")
    if clearml_task is not None:
        # Put checkpoints into hydra folders
        CHECKPOINTS_DPATH = os.path.join(os.getcwd(), checkpoint_dname)
    else:
        # User common directory for debug
        CHECKPOINTS_DPATH = os.path.join(PROJECT_ROOT, checkpoint_dname)
    os.makedirs(CHECKPOINTS_DPATH, exist_ok=True)

    model = construct_model(OmegaConf.to_container(cfg.model, resolve=True))

    optimizer = torch.optim.RAdam(lr=cfg.lr, params=model.parameters())
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer, patience=10, factor=0.2, min_lr=1.0e-7, verbose=True
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=20, T_mult=2, eta_min=1e-7, last_epoch=-1
    )

    log_metrics = ["loss", "loss_obj", "loss_box", "loss_cls"]
    batch_size = cfg.batch_size

    class CustomRunner(dl.SupervisedRunner):
        def on_loader_start(self, runner: dl.IRunner):
            super().on_loader_start(runner)
            self.meters = {
                key: metrics.AdditiveMetric(compute_on_call=False) for key in log_metrics
            }

        def handle_batch(self, batch) -> None:
            image_t, annots_t = batch["features"], batch["targets"]

            preds_t = self.model(image_t)

            main_loss, loss_parts = model.loss(preds_t, annots_t)
            if torch.isnan(main_loss):
                raise ValueError("Train Loss turned nan")

            self.batch = {
                "predictions": preds_t,
                "targets": annots_t,
            }

            self.batch_metrics.update({"loss": main_loss})
            for loss_part_name, loss_part_value in loss_parts.items():
                self.batch_metrics.update({f"loss_{loss_part_name}": loss_part_value})

            # Update epoch metrics
            for key in log_metrics:
                if key not in self.batch_metrics:
                    self.meters[key].update(0, batch_size)
                    continue

                self.meters[key].update(self.batch_metrics[key].item(), batch_size)

        def on_loader_end(self, runner):
            for key in log_metrics:
                self.loader_metrics[key] = self.meters[key].compute()[0]
            super().on_loader_end(runner)

        def on_epoch_end(self, runner):
            super().on_epoch_end(runner)

    runner = CustomRunner(input_key="features", output_key="heatmaps", target_key="targets")

    # TODO prepare linked variant
    preproc_ops = [
        LetterboxingOp(target_sz=cfg.model.infer_sz_hw),
        BboxValidationOp(),
    ]

    full_preproc_ops = [*preproc_ops, Image2TensorOp(scale_div=255)]

    # Save config to checkpoint
    checkpoint_save_dict = dict(
        config=OmegaConf.to_container(cfg, resolve=True), preprocessing=full_preproc_ops
    )
    checkpoints_suffix = f"_{clearml_task.id}" if clearml_task is not None else ""
    logger.info(f"Custom checkpoints suffix: {checkpoints_suffix}")

    additional_callbacks = []
    # additional_callbacks.append(
    #     cb.LogBestCheckpoint2ClearMLCallback(
    #         logdir=CHECKPOINTS_DPATH,
    #         loader_key="valid",
    #         metric_key="loss",
    #         minimize=True,
    #         save_kwargs=checkpoint_save_dict,
    #         clearml_task=clearml_task,
    #     )
    # )
    additional_callbacks.append(
        cb.CustomCheckpointCallback(
            logdir=CHECKPOINTS_DPATH,
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            topk=3,
            save_dict=checkpoint_save_dict,
        )
    )

    runner.train(
        model=model,
        # criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=get_loaders(cfg, preproc_ops),
        num_epochs=cfg.num_epochs,
        callbacks=[
            dl.OptimizerCallback(
                metric_key="loss",
                accumulation_steps=1,
                grad_clip_fn=nn.utils.clip_grad_norm_,
                grad_clip_params=dict(max_norm=2, norm_type=2),
            ),
            dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            *additional_callbacks,
        ],
        logdir=LOGS_DPATH,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=False,
        timeit=cfg.time_profiling,
        seed=cfg.seed,
    )


class AlbuAugmentation:
    def __init__(self, **config):
        self.fill_value = config.get("fill_value", 127)
        self.min_area_px = config.get("min_area_px", 10)
        self.min_visibility = config.get("min_visibility", 0.1)

        if not isinstance(self.fill_value, (list, tuple, np.ndarray)):
            self.fill_value = [self.fill_value] * 3

        self.description = [
            albu.ToGray(p=0.3),
            albu.OneOf(
                [
                    albu.GaussNoise(p=0.5),
                    albu.MultiplicativeNoise(per_channel=True, p=0.3),
                ],
                p=0.4,
            ),
            albu.ImageCompression(quality_lower=98, quality_upper=100, p=0.5),
            albu.OneOf(
                [
                    albu.MotionBlur(blur_limit=3, p=0.2),
                    albu.MedianBlur(blur_limit=3, p=0.2),
                    albu.GaussianBlur(blur_limit=3, p=0.2),
                    albu.Blur(blur_limit=3, p=0.2),
                ],
                p=0.2,
            ),
            albu.OneOf(
                [
                    albu.CLAHE(),
                    albu.Sharpen(),
                    albu.RandomBrightnessContrast(),
                ],
                p=0.3,
            ),
            albu.HueSaturationValue(p=0.3),
            albu.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=15,
                interpolation=3,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
                value=self.fill_value,  # Gray background
            ),
        ]
        self.compose = albu.Compose(
            self.description,
            p=1,
            bbox_params=albu.BboxParams(
                # COCO bbox ~ [ul_x, ul_y, w, h]
                format="coco",
                label_fields=["labels"],
                min_area=self.min_area_px,
                # NOTE - set visibility to filter out of border bboxes
                min_visibility=self.min_visibility,
            ),
        )

    def transform(self, img, ann):
        transformed = self.compose(image=img, bboxes=ann["bboxes"], labels=ann["labels"])

        ann["bboxes"] = np.array(transformed["bboxes"])
        ann["labels"] = np.array(transformed["labels"])
        img = transformed["image"]

        return img, ann

    def serialize(self):
        return albu.to_dict(self.compose)


if __name__ == "__main__":
    main()
