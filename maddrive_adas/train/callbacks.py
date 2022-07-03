from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
from loguru import logger

import numpy as np
import torch
from catalyst import dl
from catalyst.callbacks.checkpoint import ICheckpointCallback, save_checkpoint
from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.core.callback import CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.extras.metric_handler import MetricHandler
from catalyst.metrics._classification import MulticlassStatisticsMetric
from catalyst.metrics.functional._classification import precision, recall
from catalyst.settings import SETTINGS
from catalyst.utils import get_device, pack_checkpoint
from catalyst.utils.distributed import all_gather
from clearml import Task


class LogBestCheckpoint2ClearMLCallback(ICheckpointCallback):
    def __init__(
        self,
        logdir: str,
        # model selection info
        loader_key: str,
        metric_key: str,
        clearml_task: Task = None,
        minimize: bool = None,
        min_delta: float = 1e-6,
        # Additional data to save to checkpoint
        save_kwargs: dict = None,
        suffix: str = "",
    ):
        """Init."""
        super().__init__()

        if loader_key is not None or metric_key is not None:
            if loader_key is None or metric_key is None:
                raise ValueError(
                    "For checkpoint selection `CheckpointCallback` "
                    "requires both `loader_key` and `metric_key` specified."
                )
            self._use_model_selection = True
            self.minimize = minimize if minimize is not None else True  # loss-oriented selection
        else:
            self._use_model_selection = False
            self.minimize = False  # epoch-num-oriented selection

        self.logdir = logdir
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.is_better = MetricHandler(minimize=minimize, min_delta=min_delta)
        self.best_score = None

        self._save_kwargs = save_kwargs if save_kwargs is not None else dict()
        self._clearml_task = clearml_task
        self._suffix = suffix

    def _pack_checkpoint(self, runner: "IRunner"):
        checkpoint = pack_checkpoint(
            model=runner.model, **self._save_kwargs
        )
        return checkpoint

    def _save_checkpoint(
        self, runner: IRunner, checkpoint: dict, is_best: bool, is_last: bool
    ) -> str:
        logdir = Path(f"{self.logdir}/")
        metric_name = self.metric_key.replace("/", "_")
        checkpoint_path = save_checkpoint(
            runner=runner,
            logdir=logdir,
            checkpoint=checkpoint,
            suffix=f"best-{self.loader_key}-{metric_name}{self._suffix}",
        )

        return checkpoint_path

    def on_epoch_end(self, runner: "IRunner") -> None:
        """
        Collects and saves checkpoint after epoch.

        Args:
            runner: current runner
        """
        if runner.engine.is_ddp and not runner.engine.is_master_process:
            return

        loader_metrics = runner.epoch_metrics[self.loader_key]
        if self.metric_key not in loader_metrics:
            return

        if self._use_model_selection:
            # score model based on the specified metric
            score = runner.epoch_metrics[self.loader_key][self.metric_key]
        else:
            # score model based on epoch number
            score = runner.global_epoch_step

        is_best = False
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            is_best = True

        if not is_best:
            # Save only best!
            return

        # pack checkpoint
        checkpoint = self._pack_checkpoint(runner)
        # save checkpoint
        checkpoint_path = self._save_checkpoint(
            runner=runner, checkpoint=checkpoint, is_best=is_best, is_last=True
        )

        if self._clearml_task is not None:
            metric_name = self.metric_key.replace("/", "_")
            self._clearml_task.upload_artifact(
                name=f"Best {self.loader_key}-{metric_name} model",
                artifact_object=checkpoint_path,
            )


# NOTE - additional callback with custom fields saved
class CustomCheckpointCallback(dl.CheckpointCallback):
    def __init__(self, save_dict: dict, **kwargs) -> None:
        super().__init__(**kwargs)

        self._save_dict = save_dict

    def _save(self, runner: "IRunner", obj: Any, logprefix: str) -> str:
        logpath = f"{logprefix}.pth"
        if self.mode == "model":
            checkpoint = pack_checkpoint(model=obj, **self._save_dict)
        else:
            checkpoint = pack_checkpoint(**obj, **self._save_dict)
        
        save_checkpoint(checkpoint, logpath)
        return logpath
