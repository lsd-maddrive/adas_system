from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

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
from catalyst.utils import get_device
from catalyst.utils.distributed import all_gather
from clearml import Task

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm


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
        additional_info = dict(epoch=runner.stage_epoch_step)
        checkpoint = runner.engine.pack_checkpoint(
            model=runner.model, _info=additional_info, **self._save_kwargs
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
        if runner.is_infer_stage:
            return
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


# NOTE - additional ccallback with custom fields saved
class CustomCheckpointCallback(dl.CheckpointCallback):
    def __init__(self, save_dict: dict, **kwargs) -> None:
        super().__init__(**kwargs)

        self._save_dict = save_dict

    def _pack_checkpoint(self, runner: "IRunner"):
        checkpoint = runner.engine.pack_checkpoint(
            model=runner.model,
            criterion=runner.criterion,
            optimizer=runner.optimizer,
            scheduler=runner.scheduler,
            # experiment info
            run_key=runner.run_key,
            global_epoch_step=runner.global_epoch_step,
            global_batch_step=runner.global_batch_step,
            global_sample_step=runner.global_sample_step,
            # stage info
            stage_key=runner.stage_key,
            stage_epoch_step=runner.stage_epoch_step,
            stage_batch_step=runner.stage_batch_step,
            stage_sample_step=runner.stage_sample_step,
            # epoch info
            epoch_metrics={k: dict(v) for k, v in runner.epoch_metrics.items()},
            # loader info
            loader_key=runner.loader_key,
            loader_batch_step=runner.loader_batch_step,
            loader_sample_step=runner.loader_sample_step,
            # checkpointer info
            checkpointer_loader_key=self.loader_key,
            checkpointer_metric_key=self.metric_key,
            checkpointer_minimize=self.minimize,
            **self._save_dict,
        )
        return checkpoint


# METRICS CALCULATION CALLBACKS


def fbeta_score(precision_value, recall_value, beta=1, eps=1e-5):
    """Calculating F1-score from precision and recall to reduce computation redundancy.

    Args:
        precision_value: precision (0-1)
        recall_value: recall (0-1)
        eps: epsilon to use

    Returns:
        F-bneta score (0-1)
    """
    numerator = (1 + beta ** 2) * (precision_value * recall_value)
    denominator = beta ** 2 * precision_value + recall_value + eps
    return numerator / denominator


def get_aggregated_metrics(
    tp: np.array,
    fp: np.array,
    fn: np.array,
    support: np.array,
    zero_division: int = 0,
    beta_value: int = 1,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Count precision, recall, f1 scores per-class and with macro, weighted and micro average
    with statistics.

    Args:
        tp: array of shape (num_classes, ) of true positive statistics per class
        fp: array of shape (num_classes, ) of false positive statistics per class
        fn: array of shape (num_classes, ) of false negative statistics per class
        support: array of shape (num_classes, ) of samples count per class
        zero_division: int value, should be one of 0 or 1;
            used for precision and recall computation

    Returns:
        arrays of metrics: per-class, micro, macro, weighted averaging
    """
    num_classes = len(tp)
    precision_values = np.zeros(shape=(num_classes,))
    recall_values = np.zeros(shape=(num_classes,))
    fbeta_values = np.zeros(shape=(num_classes,))

    for i in range(num_classes):
        precision_values[i] = precision(tp=tp[i], fp=fp[i], zero_division=zero_division)
        recall_values[i] = recall(tp=tp[i], fn=fn[i], zero_division=zero_division)
        fbeta_values[i] = fbeta_score(
            precision_value=precision_values[i], recall_value=recall_values[i], beta=beta_value
        )

    per_class = (
        precision_values,
        recall_values,
        fbeta_values,
        support,
    )

    macro = (
        precision_values.mean(),
        recall_values.mean(),
        fbeta_values.mean(),
        None,
    )

    weight = support / support.sum()
    weighted = (
        (precision_values * weight).sum(),
        (recall_values * weight).sum(),
        (fbeta_values * weight).sum(),
        None,
    )

    micro_precision = precision(tp=tp.sum(), fp=fp.sum(), zero_division=zero_division)
    micro_recall = recall(tp=tp.sum(), fn=fn.sum(), zero_division=zero_division)
    micro = (
        micro_precision,
        micro_recall,
        fbeta_score(precision_value=micro_precision, recall_value=micro_recall, beta=beta_value),
        None,
    )
    return per_class, micro, macro, weighted


class MulticlassPrecisionRecallFBetaSupportMetric(MulticlassStatisticsMetric):
    """
    Metric that can collect statistics and count precision, recall, f1_score and support with it.

    Args:
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        compute_on_call: if True, allows compute metric's value on call
        compute_per_class_metrics: boolean flag to compute per-class metrics
            (default: SETTINGS.compute_per_class_metrics or False).
        prefix: metrics prefix
        suffix: metrics suffix
        num_classes: number of classes

    """

    def __init__(
        self,
        zero_division: int = 0,
        compute_on_call: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: str = None,
        suffix: str = None,
        num_classes: Optional[int] = None,
        f_beta: int = 1,
    ) -> None:
        """Init PrecisionRecallF1SupportMetric instance"""
        super().__init__(
            compute_on_call=compute_on_call, prefix=prefix, suffix=suffix, num_classes=num_classes
        )
        self.compute_per_class_metrics = compute_per_class_metrics
        self.zero_division = zero_division
        self.num_classes = num_classes
        self.f_beta = f_beta
        self.reset()

    def _convert_metrics_to_kv(self, per_class, micro, macro, weighted) -> Dict[str, float]:
        """
        Convert metrics aggregation to key-value format

        Args:
            per_class: per-class metrics, array of shape (4, self.num_classes)
                of precision, recall, f1 and support metrics
            micro: micro averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics
            macro: macro averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics
            weighted: weighted averaged metrics, array of shape (self.num_classes)
                of precision, recall, f1 and support metrics

        Returns:
            dict of key-value metrics

        """
        kv_metrics = {}
        for aggregation_name, aggregated_metrics in zip(
            ("_micro", "_macro", "_weighted"), (micro, macro, weighted)
        ):
            metrics = {
                f"{metric_name}/{aggregation_name}": metric_value
                for metric_name, metric_value in zip(
                    ("precision", "recall", f"f{self.f_beta}"), aggregated_metrics[:-1]
                )
            }
            kv_metrics.update(metrics)

        # @TODO: rewrite this block - should be without `num_classes`
        if self.compute_per_class_metrics:
            per_class_metrics = {
                f"{metric_name}/class_{i:02d}": metric_value[i]
                for metric_name, metric_value in zip(
                    ("precision", "recall", f"f{self.f_beta}", "support"), per_class
                )
                for i in range(self.num_classes)
            }
            kv_metrics.update(per_class_metrics)
        return kv_metrics

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[Any, Any, Any, Any]:
        """
        Update statistics and return intermediate metrics results

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            tuple of metrics intermediate results with per-class, micro, macro and
                weighted averaging

        """
        tn, fp, fn, tp, support, num_classes = super().update(outputs=outputs, targets=targets)
        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=tp,
            fp=fp,
            fn=fn,
            support=support,
            zero_division=self.zero_division,
            beta_value=self.f_beta,
        )
        if self.num_classes is None:
            self.num_classes = num_classes

        return per_class, micro, macro, weighted

    def update_key_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update statistics and return intermediate metrics results

        Args:
            outputs: prediction values
            targets: true answers

        Returns:
            dict of metrics intermediate results

        """
        per_class, micro, macro, weighted = self.update(outputs=outputs, targets=targets)
        metrics = self._convert_metrics_to_kv(
            per_class=per_class, micro=micro, macro=macro, weighted=weighted
        )
        return metrics

    def compute(self) -> Any:
        """
        Compute precision, recall, f1 score and support.
        Compute micro, macro and weighted average for the metrics.

        Returns:
            list of aggregated metrics: per-class, micro, macro and weighted averaging of
                precision, recall, f1 score and support metrics

        """
        # ddp hotfix, could be done better
        # but metric must handle DDP on it's own
        if self._ddp_backend == "xla":
            device = get_device()
            for key in self.statistics:
                key_statistics = torch.tensor([self.statistics[key]], device=device)
                key_statistics = xm.all_gather(key_statistics).sum(dim=0).cpu().numpy()
                self.statistics[key] = key_statistics
        elif self._ddp_backend == "ddp":
            for key in self.statistics:
                value: List[np.ndarray] = all_gather(self.statistics[key])
                value: np.ndarray = np.sum(np.vstack(value), axis=0)
                self.statistics[key] = value

        per_class, micro, macro, weighted = get_aggregated_metrics(
            tp=self.statistics["tp"],
            fp=self.statistics["fp"],
            fn=self.statistics["fn"],
            support=self.statistics["support"],
            zero_division=self.zero_division,
            beta_value=self.f_beta,
        )
        if self.compute_per_class_metrics:
            return per_class, micro, macro, weighted
        else:
            return [], micro, macro, weighted

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute precision, recall, fbeta score and support.
        Compute micro, macro and weighted average for the metrics.

        Returns:
            dict of metrics

        """
        per_class, micro, macro, weighted = self.compute()
        metrics = self._convert_metrics_to_kv(
            per_class=per_class, micro=micro, macro=macro, weighted=weighted
        )
        return metrics


class PrecisionRecallFbetaSupportCallback(BatchMetricCallback):
    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_classes: Optional[int] = None,
        zero_division: int = 0,
        log_on_batch: bool = True,
        compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
        prefix: str = None,
        suffix: str = None,
        f_beta: int = 1,
    ):
        """Init."""
        super().__init__(
            metric=MulticlassPrecisionRecallFBetaSupportMetric(
                zero_division=zero_division,
                prefix=prefix,
                suffix=suffix,
                compute_per_class_metrics=compute_per_class_metrics,
                num_classes=num_classes,
                f_beta=f_beta,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )
