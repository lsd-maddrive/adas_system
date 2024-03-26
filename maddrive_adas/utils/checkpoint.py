from typing import Tuple


import torch

from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler

from maddrive_adas.models.yolo import Model
from maddrive_adas.utils.general import one_cycle
from maddrive_adas.utils.torch_utils import smart_optimizer


class Checkpoint:

    MODEL = 'MODEL'
    OPTIMIZER = 'OPTIMIZER'
    SCHEDULER = 'SCHEDULER'
    EPOCH = 'EPOCH'
    _HYPS = 'HYPS'

    _MODEL_CONFIG = 'MODEL_CONFIG'
    _TOTAL_EPOCHS = 'TOTAL_EPOCHS'
    _IMG_SIZE = 'IMGSZ'

    _REQUIRED_KEYS = [
        MODEL, OPTIMIZER, SCHEDULER, EPOCH, _HYPS,
        _MODEL_CONFIG, _TOTAL_EPOCHS, _IMG_SIZE
    ]

    def __init__(self, checkpoint_file_path: str):
        """Checkpoint constructor.

        Args:
            checkpoint_path (str): Checkpoint file path.

        Raises:
            ValueError: In case keys mismatch.
        """
        try:
            self._dict: dict = torch.load(checkpoint_file_path)
            self._verify_dict()
            self._hyps: dict = self._dict[Checkpoint._HYPS]
        except FileNotFoundError:
            msg = f"File {checkpoint_file_path} doesn't exists."
            raise ValueError(msg)

    def _verify_dict(self):
        if all(x in self._dict for x in Checkpoint._REQUIRED_KEYS):
            return

        msg = f'Invalid checkpoint file. Required keys: {sorted(Checkpoint._REQUIRED_KEYS)}.'
        msg += f'Found keys: {sorted(list(self._dict.keys()))}.'
        raise ValueError(msg)

    def get_hyps(self) -> dict:
        return self._dict[Checkpoint._HYPS]

    def load_eval_checkpoint(self, map_device: torch.device) -> Model:
        model = Model(cfg=self._dict[Checkpoint._MODEL_CONFIG], ch=3, nc=1)
        model.load_state_dict(self._dict[Checkpoint.MODEL])
        model.to(map_device)
        return model.eval()

    def get_checkpoint_img_size(self) -> Tuple[int, int]:
        return self._dict[Checkpoint._IMG_SIZE]

    @staticmethod
    def build_checkpoint(
        model: Model,
        hyps: dict,
        model_config: dict,
        optimizer: Optimizer,
        scheduler,
        imgsz: int,
        initial_epoch: int,
        total_epochs: int,
        output_path: str
    ):
        torch.save({
            Checkpoint._MODEL_CONFIG: model_config,
            Checkpoint._HYPS: hyps,
            Checkpoint.EPOCH: initial_epoch,
            Checkpoint.MODEL: model.state_dict(),
            Checkpoint.OPTIMIZER: optimizer.state_dict(),
            Checkpoint.SCHEDULER: scheduler.state_dict(),
            Checkpoint._TOTAL_EPOCHS: total_epochs,
            Checkpoint._IMG_SIZE: imgsz,
        }, output_path)

    def load_train_checkpoint(self, map_device: torch.device) -> Tuple[Model, Optimizer, object, int]:
        """Load train checkpoint.

        Returns:
            Tuple[Model, Optimizer, object, int]: Model, optimizer, scheduler and last epoch.
        """
        model = self.load_eval_checkpoint(map_device=map_device)
        optimizer: Optimizer = smart_optimizer(
            model, 'SGD', self._hyps['lr0'], self._hyps['momentum'], self._hyps['weight_decay'])
        optimizer.load_state_dict(self._dict[Checkpoint.OPTIMIZER])

        # cosine 1->hyp['lrf']
        lf = one_cycle(1, self._hyps['lrf'], self._dict[Checkpoint._TOTAL_EPOCHS])
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.load_state_dict(self._dict[Checkpoint.SCHEDULER])

        return model.eval(), optimizer, scheduler, self._dict[Checkpoint.EPOCH]

    def save_checkpoint(
        self,
        model: Model,
        optimizer: Optimizer,
        scheduler: lr_scheduler.CyclicLR,
        epoch: int,
        total_epochs: int,
        output_path: str,
    ):
        Checkpoint.build_checkpoint(
            model,
            self._dict[Checkpoint._HYPS],
            self._dict[Checkpoint._MODEL_CONFIG],
            optimizer=optimizer,
            scheduler=scheduler,
            imgsz=self.get_checkpoint_img_size(),
            initial_epoch=epoch,
            total_epochs=total_epochs,
            output_path=output_path
        )
