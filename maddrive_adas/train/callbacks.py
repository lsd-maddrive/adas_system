from typing import Any

from pathlib import Path

from catalyst import dl
from catalyst.callbacks.checkpoint import ICheckpointCallback, save_checkpoint
from catalyst.core.runner import IRunner
from catalyst.extras.metric_handler import MetricHandler

from catalyst.utils import pack_checkpoint

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
        checkpoint = pack_checkpoint(model=runner.model, **self._save_kwargs)
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
        logpath = f"{logprefix}-{self.metric_key}.pth"
        if self.mode == "model":
            checkpoint = pack_checkpoint(model=obj, **self._save_dict)
        else:
            checkpoint = pack_checkpoint(**obj, **self._save_dict)

        save_checkpoint(checkpoint, logpath)
        return logpath


# class MAPCallback(dl.BatchMetricCallback):
#     def __init__(
#         self,
#         input_key: Union[str, Iterable[str], Dict[str, str]],
#         target_key: Union[str, Iterable[str], Dict[str, str]],
#         class_dim: int = 1,
#         weights: Optional[List[float]] = None,
#         class_names: Optional[List[str]] = None,
#         threshold: Optional[float] = None,
#         log_on_batch: bool = True,
#         compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
#         prefix: str = None,
#         suffix: str = None,
#     ) -> None:
#         """Init."""
#         super().__init__(
#             metric=MAPMetric(
#                 class_dim=class_dim,
#                 weights=weights,
#                 class_names=class_names,
#                 threshold=threshold,
#                 compute_per_class_metrics=compute_per_class_metrics,
#                 prefix=prefix,
#                 suffix=suffix,
#             ),
#             input_key=input_key,
#             target_key=target_key,
#             log_on_batch=log_on_batch,
#         )


# from catalyst.utils.distributed import all_gather, get_backend


# class MAPMetric(ICallbackBatchMetric):
#     def __init__(
#         self,
#         metric_fn: Callable,
#         metric_name: str,
#         class_dim: int = 1,
#         weights: Optional[List[float]] = None,
#         class_names: Optional[List[str]] = None,
#         threshold: Optional[float] = 0.5,
#         compute_on_call: bool = True,
#         compute_per_class_metrics: bool = SETTINGS.compute_per_class_metrics,
#         prefix: Optional[str] = None,
#         suffix: Optional[str] = None,
#     ):
#         """Init"""
#         super().__init__(compute_on_call, prefix, suffix)
#         self.metric_fn = metric_fn
#         self.metric_name = metric_name
#         self.class_dim = class_dim
#         self.threshold = threshold
#         self.compute_per_class_metrics = compute_per_class_metrics
#         # statistics = {class_idx: {"tp":, "fn": , "fp": "tn": }}
#         self.statistics = {}
#         self.weights = weights
#         self.class_names = class_names
#         self._checked_params = False
#         self._ddp_backend = None

#     def _check_parameters(self):
#         # check class_names
#         # TODO

#         if self.class_names is not None:
#             assert len(self.class_names) == len(self.statistics), (
#                 f"the number of class names must be equal to the number of classes,"
#                 " got weights"
#                 f" {len(self.class_names)} and classes: {len(self.statistics)}"
#             )
#         else:
#             self.class_names = [
#                 f"class_{idx:02d}" for idx in range(len(self.statistics))
#             ]
#         if self.weights is not None:
#             assert len(self.weights) == len(self.statistics), (
#                 f"the number of weights must be equal to the number of classes,"
#                 " got weights"
#                 f" {len(self.weights)} and classes: {len(self.statistics)}"
#             )

#     def reset(self):
#         """Reset all statistics"""
#         self.statistics = {}
#         self._ddp_backend = get_backend()

#     def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """Updates segmentation statistics with new data
#         and return intermediate metrics values.

#         Args:
#             outputs: tensor of logits
#             targets: tensor of targets

#         Returns:
#             metric for each class
#         """
#         tp, fp, fn = get_segmentation_statistics(
#             outputs=outputs.cpu().detach(),
#             targets=targets.cpu().detach(),
#             class_dim=self.class_dim,
#             threshold=self.threshold,
#         )

#         for idx, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn)):
#             if idx in self.statistics:
#                 self.statistics[idx]["tp"] += tp_class
#                 self.statistics[idx]["fp"] += fp_class
#                 self.statistics[idx]["fn"] += fn_class
#             else:
#                 self.statistics[idx] = {}
#                 self.statistics[idx]["tp"] = tp_class
#                 self.statistics[idx]["fp"] = fp_class
#                 self.statistics[idx]["fn"] = fn_class

#         # need only one time
#         if not self._checked_params:
#             self._check_parameters()
#             self._checked_params = True

#         metrics_per_class = self.metric_fn(tp, fp, fn)
#         return metrics_per_class

#     def update_key_value(
#         self, outputs: torch.Tensor, targets: torch.Tensor
#     ) -> Dict[str, torch.Tensor]:
#         """Updates segmentation statistics with new data
#         and return intermediate metrics values.

#         Args:
#             outputs: tensor of logits
#             targets: tensor of targets

#         Returns:
#             dict of metric for each class and weighted (if weights were given) metric
#         """
#         metrics_per_class = self.update(outputs, targets)
#         macro_metric = torch.mean(metrics_per_class)
#         metrics = {
#             f"{self.prefix}{self.metric_name}{self.suffix}/{self.class_names[idx]}": val
#             for idx, val in enumerate(metrics_per_class)
#         }
#         metrics[f"{self.prefix}{self.metric_name}{self.suffix}"] = macro_metric
#         if self.weights is not None:
#             weighted_metric = 0
#             for idx, value in enumerate(metrics_per_class):
#                 weighted_metric += value * self.weights[idx]
#             metrics[
#                 f"{self.prefix}{self.metric_name}{self.suffix}/_weighted"
#             ] = weighted_metric
#         return metrics

#     def compute(self):
#         """
#         Compute metrics with accumulated statistics

#         Returns:
#             tuple of metrics: per_class, micro_metric, macro_metric,
#                 weighted_metric(None if weights is None)
#         """
#         per_class = []
#         total_statistics = {}
#         macro_metric = 0
#         weighted_metric = 0
#         # ddp hotfix, could be done better
#         # but metric must handle DDP on it's own
#         # TODO: optimise speed
#         if self._ddp_backend == "xla":
#             device = get_device()
#             for _, statistics in self.statistics.items():
#                 for key in statistics:
#                     value = torch.tensor([statistics[key]], device=device)
#                     statistics[key] = xm.all_gather(value).sum(dim=0)
#         elif self._ddp_backend == "ddp":
#             for _, statistics in self.statistics.items():
#                 for key in statistics:
#                     value: List[torch.Tensor] = all_gather(statistics[key])
#                     value: torch.Tensor = torch.sum(torch.vstack(value), dim=0)
#                     statistics[key] = value

#         for class_idx, statistics in self.statistics.items():
#             value = self.metric_fn(**statistics)
#             per_class.append(value)
#             macro_metric += value
#             if self.weights is not None:
#                 weighted_metric += value * self.weights[class_idx]
#             for stats_name, value in statistics.items():
#                 total_statistics[stats_name] = (
#                     total_statistics.get(stats_name, 0) + value
#                 )

#         macro_metric /= len(self.statistics)
#         micro_metric = self.metric_fn(**total_statistics)

#         if self.weights is None:
#             weighted_metric = None
#         if self.compute_per_class_metrics:
#             return per_class, micro_metric, macro_metric, weighted_metric
#         else:
#             return [], micro_metric, macro_metric, weighted_metric

#     def compute_key_value(self) -> Dict[str, torch.Tensor]:
#         """
#         Compute segmentation metric for all data and return results in key-value format

#         Returns:
#              dict of metrics, including micro, macro
#                 and weighted (if weights were given) metrics
#         """
#         per_class, micro_metric, macro_metric, weighted_metric = self.compute()

#         metrics = {}
#         for class_idx, value in enumerate(per_class):
#             class_name = self.class_names[class_idx]
#             metrics[f"{self.prefix}{self.metric_name}{self.suffix}/{class_name}"] = value

#         metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_micro"] = micro_metric
#         metrics[f"{self.prefix}{self.metric_name}{self.suffix}"] = macro_metric
#         metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_macro"] = macro_metric
#         if self.weights is not None:
#             # @TODO: rename this one
#             metrics[
#                 f"{self.prefix}{self.metric_name}{self.suffix}/_weighted"
#             ] = weighted_metric
#         return metrics
