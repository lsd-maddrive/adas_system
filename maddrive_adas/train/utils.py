
import hashlib
import json
import logging
import os
import pydoc

import torch
from catalyst import utils as cata_ut

logger = logging.getLogger(__name__)


def construct_model(d: dict, **default_kwargs: dict):
    COMMON_PATH = "maddrive_adas.sign_det.models"

    assert isinstance(d, dict) and "type" in d
    kwargs = d.copy()
    model_type = kwargs.pop("type")

    if "Yolo4" in model_type:
        model_type = f"{COMMON_PATH}.yolo4.{model_type}"

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    constructor = pydoc.locate(model_type)
    if constructor is None:
        raise NotImplementedError(f"Model {model_type} not found")

    return constructor(**kwargs)



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
