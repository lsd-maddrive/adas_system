from typing import Tuple

import yaml

import pytest
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer

from maddrive_adas.utils import get_project_root
from maddrive_adas.models.yolo import Model
from maddrive_adas.utils.checkpoint import Checkpoint
from maddrive_adas.utils.general import one_cycle
from maddrive_adas.utils.torch_utils import smart_optimizer


PROJECT_ROOT = get_project_root()
TEST_BUILD_CHECKPOINT_PATH = './test'
INITIAL_EPOCH = 3
TOTAL_EPOCHS = 300
IMAGE_SIZE = 412


@pytest.fixture
def checkpoint_test_data_path():
    return get_project_root() / 'tests' / 'test_data' / 'test_checkpoint_data'


@pytest.fixture
def model_config(checkpoint_test_data_path):
    with open(checkpoint_test_data_path / 'yolov5m_custom_anchors.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def hyps(checkpoint_test_data_path):
    with open(checkpoint_test_data_path / 'hyp.scratch.yaml') as f:
        return yaml.safe_load(f)


@pytest.mark.checkpoint
def test_checkpoint_constructor_invalid_path():
    try:
        Checkpoint('')
    except ValueError:
        pass


@pytest.mark.checkpoint
def test_build_checkpoint(model_config, hyps):
    model, optimizer, scheduler = create_sample_model_optimizer_scheduler(
        model_config, hyps
    )

    Checkpoint.build_checkpoint(
        model=model,
        hyps=hyps,
        model_config=model_config,
        scheduler=scheduler,
        optimizer=optimizer,
        imgsz=IMAGE_SIZE,
        initial_epoch=INITIAL_EPOCH,
        total_epochs=TOTAL_EPOCHS,
        output_path=TEST_BUILD_CHECKPOINT_PATH,
    )


def create_sample_model_optimizer_scheduler(
    model_config,
    hyps
) -> Tuple[Model, Optimizer, object]:
    model = Model(model_config, ch=3, nc=1)
    optimizer: Optimizer = smart_optimizer(
        model, 'SGD', hyps['lr0'], hyps['momentum'], hyps['weight_decay'])
    lf = one_cycle(1, hyps['lrf'], )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return model, optimizer, scheduler


@pytest.mark.checkpoint
def test_load_eval_checkpoint(model_config, hyps):
    test_build_checkpoint(model_config, hyps)
    checkpoint = Checkpoint(TEST_BUILD_CHECKPOINT_PATH)
    model = checkpoint.load_eval_checkpoint(map_device='cpu')

    assert isinstance(model, Model)


@pytest.mark.checkpoint
def test_load_train_checkpoint(model_config, hyps):
    test_build_checkpoint(model_config, hyps)
    checkpoint = Checkpoint(TEST_BUILD_CHECKPOINT_PATH)
    model, optimizer, scheduler, epoch = checkpoint.load_train_checkpoint(
        map_device='cpu'
    )

    assert isinstance(model, Model)
    assert isinstance(optimizer, Optimizer)
    assert isinstance(scheduler, lr_scheduler.LambdaLR)
    assert epoch == INITIAL_EPOCH


@pytest.mark.checkpoint
def test_save_load_checkpoint(model_config, hyps):
    SAMPLE_EPOCH_NUM = 27

    test_build_checkpoint(model_config, hyps)
    initial_checkpoint = Checkpoint(TEST_BUILD_CHECKPOINT_PATH)
    initial_model, optimizer, scheduler, epoch = initial_checkpoint.load_train_checkpoint(
        map_device='cpu'
    )

    initial_checkpoint.save_checkpoint(
        model=initial_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=SAMPLE_EPOCH_NUM,
        total_epochs=TOTAL_EPOCHS,
        output_path=TEST_BUILD_CHECKPOINT_PATH
    )
    del optimizer, scheduler, epoch, initial_checkpoint

    checkpoint = Checkpoint(TEST_BUILD_CHECKPOINT_PATH)
    loaded_model, loaded_optimizer, loaded_scheduler, loaded_epoch = \
        checkpoint.load_train_checkpoint(
            map_device='cpu'
        )

    assert loaded_epoch == SAMPLE_EPOCH_NUM
    assert is_model_weight_equal(initial_model, loaded_model)


def is_model_weight_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_cuda_load(model_config, hyps):
    test_build_checkpoint(model_config, hyps)
    checkpoint = Checkpoint(TEST_BUILD_CHECKPOINT_PATH)
    model = checkpoint.load_eval_checkpoint(map_device='cuda')

    assert next(model.parameters()).is_cuda
