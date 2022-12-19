from maddrive_adas.utils.general import LOGGER
from typing import Tuple

import pathlib
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from maddrive_adas.utils import is_debug
from torch.optim import lr_scheduler
from maddrive_adas.models.yolo import Model
from maddrive_adas.utils.winter_augment import WinterizedYoloDataset
from maddrive_adas.utils.datasets import YoloDataset
from maddrive_adas.val import valid_epoch
from maddrive_adas.utils.general import one_cycle, set_logging
from maddrive_adas.utils.torch_utils import smart_optimizer

PROJECT_ROOT = pathlib.Path('./').resolve()
DATA_DIR = PROJECT_ROOT / 'SignDetectorAndClassifier' / 'data'
DATASET_DIR = DATA_DIR / 'YOLO_DATASET'
CHECKPOINT_PATH = DATA_DIR / 'YOLO_CP.pt'
HYP_YAML_FILE_PATH = DATA_DIR / "hyp.scratch.yaml"
MODEL_CONFIG_PATH = DATA_DIR / 'yolov5s_custom_anchors.yaml'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMGSZ = 640
TOTAL_EPOCHS = 300
CURRENT_EPOCH = -1


def read_hyp(hyps_file):
    import yaml
    with open(hyps_file, errors='ignore') as f:
        return yaml.safe_load(f)


HYP = read_hyp(hyps_file=HYP_YAML_FILE_PATH)


def train(
    model: Model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs,
    hyp,
    optimizer: Optimizer = None,
    scheduler=None,
    start_epoch: int = 0,
    half=False,
):
    from maddrive_adas.utils.loss import ComputeLoss
    from maddrive_adas.utils.torch_utils import ModelEMA, de_parallel

    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    from torch.cuda import amp

    global CURRENT_EPOCH
    nc = 1
    accumulate_not_initialized = True

    cuda = DEVICE.type == 'cuda'
    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    nbs = 64  # nominal batch size
    batch_size = train_loader.batch_size
    last_opt_step = -1

    if not optimizer:
        LOGGER.info('Optimizer not passed. Using default Yolo\'s SDG')
        optimizer: Optimizer = smart_optimizer(
            model, 'SDG', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    if not scheduler:
        LOGGER.info('Scheduler not passed. Using default CyclicLR')
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        lf = scheduler.lr_lambdas[0]

    ema = ModelEMA(model)

    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (IMGSZ / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model

    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)

    LOGGER.info('Starting model training')
    for epoch in range(start_epoch, epochs):
        CURRENT_EPOCH = epoch

        model.train()

        mloss = torch.zeros(3, device=DEVICE)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem',
                    'box', 'obj', 'cls', 'labels', 'img_size'))
        pbar = tqdm(enumerate(train_loader), total=nb,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

        optimizer.zero_grad()
        for i, (imgs, targets, paths, original_boxes) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(DEVICE, non_blocking=True).float() / \
                255  # uint8 to float32, 0-255 to 0.0-1.0
            imgs.half() if half else None
            # writer.add_image_with_boxes('sample img', imgs[0], targets[0], epoch)
            # writer.add_images('images batch', imgs, epoch)

            # Warmup
            if ni <= nw or accumulate_not_initialized:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j ==
                                        2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
                accumulate_not_initialized = False

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(DEVICE))  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        writer.add_scalar('lr0', lr[0], epoch)
        writer.add_scalar('lr1', lr[1], epoch)
        writer.add_scalar('lr2', lr[2], epoch)
        writer.add_scalar('loss box', mloss[0], epoch)
        writer.add_scalar('loss_obj', mloss[1], epoch)

        mp, mr, map50, map_, maps = valid_epoch(
            model,
            valid_loader,
            conf_thres=0.3,
            iou_thres=0.4,
            max_det=10,
            half=half
        )
        writer.add_scalars(
            'Metrics',
            {
                'mp': mp,
                'mr': mr,
                'map50': map50,
                'map': map_,
                'precision': maps[0]
            }, epoch)
        model.float()

        # writer.add_hparams(
        #     {
        #         'epoch': epoch,
        #     },
        #     {
        #         'map': .0,  # TODO: fix map
        #         'lbox': mloss[0],
        #         'lobj': mloss[1],
        #         'lr_1': lr[0],
        #         'lr_2': lr[1],
        #         'lr_3': lr[2]
        #     },)

        writer.flush()

        scheduler.step()

        now = datetime.now()
        model_per_iter_name = f'YoloV5_{now.strftime("%d.%m_%H.%M")}_lbox{mloss[0]}_lobj{mloss[1]}.pt'
        model_save_name = DATA_DIR / model_per_iter_name

        # Checkpoint.save_checkpoint(model_save_name, model, optimizer, scheduler, epoch)
        Checkpoint.save_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, epoch)


def load_dataset_csv() -> pd.DataFrame:
    import ast

    def read_yolo_dataset_csv(csv_path: pathlib.Path, filepath_prefix: str):
        data = pd.read_csv(csv_path) if not is_debug() else pd.read_csv(csv_path).iloc[:200]
        data['filepath'] = data['filepath'].apply(lambda x: pathlib.Path(filepath_prefix) / x)
        data['size'] = data['size'].apply(lambda x: ast.literal_eval(x))
        data['coords'] = data['coords'].apply(lambda x: ast.literal_eval(x))
        return data

    dataset = read_yolo_dataset_csv(
        csv_path=DATASET_DIR / 'USER_FULL_FRAMES.csv',
        filepath_prefix=DATASET_DIR
    )

    return dataset


def get_winterized_dataloader(
    df: pd.DataFrame,
    set_label: str,
    hyp,
    imgsz,
    batch_size=1,
    num_workers=0,
    augment=False,
):
    dataset = WinterizedYoloDataset(
        df,
        set_label=set_label,
        hyp_arg=hyp,
        img_size=imgsz,
        augment=augment,
        hide_corner_chance=1.,
    )
    # dataset = YoloDataset(
    #     df,
    #     set_label=set_label,
    #     hyp_arg=hyp,
    #     augment=augment,
    #     batch_size=batch_size,
    #     img_size=imgsz)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # doesnt work in Windows
        sampler=None,
        pin_memory=True,
        collate_fn=WinterizedYoloDataset.collate_fn,
    )


class Checkpoint:

    MODEL = 'MODEL'
    OPTIMIZER = 'OPTIMIZER'
    SCHEDULER = 'SCHEDULER'
    EPOCH = 'EPOCH'

    @staticmethod
    def load_checkpoint(path, model_config_path, map_location, hyp) -> Tuple[Model, Optimizer, object]:
        LOGGER.info(f'Loading checkpoint from {path}')
        # channels = 3 - img depth, nc = 1 - number classes
        model = Model(cfg=model_config_path, ch=3, nc=1)
        optimizer: Optimizer = smart_optimizer(
            model, 'SGD', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

        lf = one_cycle(1, hyp['lrf'], TOTAL_EPOCHS)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        try:
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint[Checkpoint.MODEL])
            model.eval()
            model.to(map_location)

            optimizer.load_state_dict(checkpoint[Checkpoint.OPTIMIZER])
            scheduler.load_state_dict(checkpoint[Checkpoint.SCHEDULER])
            LOGGER.info('Checkpoint loaded successfully')
            return model, optimizer, scheduler, checkpoint[Checkpoint.EPOCH]
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            msg = f'Unable to load model checkpoint: {e}'
            LOGGER.warning(msg)
            model.to(map_location)
            return model, optimizer, scheduler, 0

    @staticmethod
    def save_checkpoint(
        path,
        model: Model,
        optimizer: Optimizer,
        scheduler: lr_scheduler.CyclicLR,
        epoch: int
    ):
        torch.save({
            Checkpoint.EPOCH: epoch,
            Checkpoint.MODEL: model.state_dict(),
            Checkpoint.OPTIMIZER: optimizer.state_dict(),
            Checkpoint.SCHEDULER: scheduler.state_dict()
        }, path)
        LOGGER.info('Model saved')


if __name__ == '__main__':
    full_dataset = load_dataset_csv()
    train_loader = get_winterized_dataloader(
        full_dataset, set_label='train', hyp=HYP, batch_size=32, imgsz=IMGSZ)
    valid_loader = get_winterized_dataloader(
        full_dataset, set_label='valid', hyp=HYP, batch_size=64, imgsz=IMGSZ)

    model, optimizer, scheduler, epoch = Checkpoint.load_checkpoint(
        CHECKPOINT_PATH,
        MODEL_CONFIG_PATH,
        hyp=HYP,
        map_location=DEVICE
    )

    try:
        train(
            model, train_loader,
            valid_loader, start_epoch=epoch,
            epochs=300, hyp=HYP,
            optimizer=optimizer,
            scheduler=scheduler,
            half=False)
    except KeyboardInterrupt:
        LOGGER.warning('KeyboardInterrupt occur. Saving model.')
        Checkpoint.save_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, CURRENT_EPOCH)
