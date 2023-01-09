import ast
import yaml
import pathlib

import torch
import numpy as np
import pandas as pd

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from maddrive_adas.utils.general import one_cycle, LOGGER
from torch.utils.tensorboard import SummaryWriter

from maddrive_adas.utils import is_debug, get_project_root
from maddrive_adas.models.yolo import Model
from maddrive_adas.utils.winter_augment import WinterizedYoloDataset
from maddrive_adas.val import valid_epoch
from maddrive_adas.utils.checkpoint import Checkpoint
from maddrive_adas.utils.torch_utils import smart_optimizer


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / 'SignDetectorAndClassifier' / 'data'
DATASET_DIR = DATA_DIR / 'YOLO_DATASET'
CHECKPOINT_PATH = DATA_DIR / 'YOLO_CP_m.pt'
HYP_YAML_FILE_PATH = DATA_DIR / "hyp.scratch.yaml"
MODEL_CONFIG_PATH = DATA_DIR / 'yolov5m_custom_anchors.yaml'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMGSZ = 640
TOTAL_EPOCHS = 300
CURRENT_EPOCH = -1
CONF_THRES = 0.3
IOU_THRES = 0.4


def train(
    model: Model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    total_epochs,
    hyp,
    optimizer: Optimizer,
    scheduler: lr_scheduler.CyclicLR,
    checkpoint: Checkpoint,
    start_epoch,
    half,
):
    from maddrive_adas.utils.loss import ComputeLoss
    from maddrive_adas.utils.torch_utils import ModelEMA, de_parallel

    from tqdm import tqdm
    writer = SummaryWriter()
    from torch.cuda import amp

    global CURRENT_EPOCH
    nc = 1

    cuda = DEVICE.type == 'cuda'
    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    nbs = 64  # nominal batch size
    batch_size = train_loader.batch_size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    last_opt_step = -1

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
    for epoch in range(start_epoch, total_epochs):
        CURRENT_EPOCH = epoch

        model.train()
        mloss = torch.zeros(3, device=DEVICE)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem',
                    'box', 'obj', 'cls', 'labels', 'img_size'))
        pbar = tqdm(enumerate(train_loader), total=nb,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=True)  # progress bar

        optimizer.zero_grad()
        for i, (imgs, targets, paths, original_images_shapes) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(DEVICE, non_blocking=True).float() / \
                255  # uint8 to float32, 0-255 to 0.0-1.0
            imgs.half() if half else None

            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(
                    1, np.interp(ni, xi, [1, nbs / batch_size]).round()
                )
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    if j == 2:
                        fp_ = hyp['warmup_bias_lr']
                    else:
                        fp_ = 0.0
                    x['lr'] = np.interp(
                        ni, xi,
                        [fp_, x['initial_lr'] * scheduler.lr_lambdas[0](epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

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
                ema.update(model)
                last_opt_step = ni

            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                f'{epoch}/{total_epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers

        mp, mr, map50, map_, maps = valid_epoch(
            model,
            valid_loader,
            conf_thres=CONF_THRES,
            iou_thres=IOU_THRES,
            max_det=10,
            half=half,
            writer=writer,
            epoch=epoch,
            writer_limit=10
        )

        write_hparams(writer, lr, mloss, epoch=epoch)
        write_metrics(writer, metrics={
            'mp': mp,
            'mr': mr,
            'map50': map50,
            'map': map_,
            'precision': maps[0]
        }, epoch=epoch)

        model.float()
        writer.flush()

        scheduler.step()

        checkpoint.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            total_epochs=total_epochs,
            output_path=CHECKPOINT_PATH,
        )


def load_dataset_csv() -> pd.DataFrame:
    dataset = read_yolo_dataset_csv(
        csv_path=DATASET_DIR / 'USER_FULL_FRAMES.csv',
        filepath_prefix=DATASET_DIR
    )

    return dataset


def read_yolo_dataset_csv(csv_path: pathlib.Path, filepath_prefix: str):
    data = pd.read_csv(csv_path).iloc[:400] if is_debug() else pd.read_csv(csv_path)
    data['filepath'] = data['filepath'].apply(lambda x: pathlib.Path(filepath_prefix) / x)
    data['size'] = data['size'].apply(lambda x: ast.literal_eval(x))
    data['coords'] = data['coords'].apply(lambda x: ast.literal_eval(x))
    return data


def write_hparams(writer: SummaryWriter, lr, mloss, epoch):
    writer.add_scalar('lr weight with decay', lr[0], epoch)
    writer.add_scalar('lr weights', lr[1], epoch)
    writer.add_scalar('lr biases', lr[2], epoch)
    writer.add_scalar('loss box', mloss[0], epoch)
    writer.add_scalar('loss_obj', mloss[1], epoch)


def write_metrics(writer: SummaryWriter, metrics, epoch):
    writer.add_scalars(
        'Metrics', metrics, global_step=epoch)


def get_winterized_dataloader(
    df: pd.DataFrame,
    set_label: str,
    hyp,
    imgsz,
    batch_size=1,
    num_workers=0,
    augment=False,
    shuffle=False,
):
    dataset = WinterizedYoloDataset(
        df,
        set_label=set_label,
        hyp_arg=hyp,
        img_size=imgsz,
        augment=augment,
        hide_corner_chance=1.,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # doesnt work in Windows
        sampler=None,
        pin_memory=True,
        collate_fn=WinterizedYoloDataset.collate_fn,
    )


def get_or_load_checkpoint(checkpoint_file_path) -> Checkpoint:
    try:
        return _load_checkpoint(checkpoint_file_path=checkpoint_file_path)
    except ValueError:
        _build_checkpoint(checkpoint_file_path)
        return _load_checkpoint(checkpoint_file_path)


def _load_checkpoint(checkpoint_file_path):
    return Checkpoint(checkpoint_file_path)


def _build_checkpoint(checkpoint_file_path):
    hyps = load_yaml(HYP_YAML_FILE_PATH)
    # channels = 3 - img depth, nc = 1 - number classes
    model = Model(cfg=MODEL_CONFIG_PATH, ch=3, nc=1)
    optimizer: Optimizer = smart_optimizer(
        model, 'SGD', hyps['lr0'], hyps['momentum'], hyps['weight_decay'])

    lf = one_cycle(1, hyps['lrf'], TOTAL_EPOCHS)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    Checkpoint.build_checkpoint(
        model=model,
        hyps=hyps,
        model_config=model.yaml,
        optimizer=optimizer,
        scheduler=scheduler,
        initial_epoch=0,
        total_epochs=TOTAL_EPOCHS,
        output_path=checkpoint_file_path
    )


def load_yaml(yaml_path) -> dict:
    with open(yaml_path, errors='ignore') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    full_dataset = load_dataset_csv()
    checkpoint = get_or_load_checkpoint(CHECKPOINT_PATH)
    hyps = checkpoint.get_hyps()
    train_loader = get_winterized_dataloader(
        full_dataset, set_label='train', hyp=hyps, batch_size=20,
        imgsz=IMGSZ, shuffle=True, augment=True)
    valid_loader = get_winterized_dataloader(
        full_dataset, set_label='valid', hyp=hyps, batch_size=40, imgsz=IMGSZ, shuffle=False)

    model, optimizer, scheduler, epoch = checkpoint.load_train_checkpoint(
        map_device=DEVICE
    )

    try:
        train(
            model, train_loader,
            valid_loader, start_epoch=epoch,
            total_epochs=300, hyp=hyps,
            optimizer=optimizer,
            scheduler=scheduler,
            half=True, checkpoint=checkpoint)
    except KeyboardInterrupt:
        LOGGER.warning('KeyboardInterrupt occur. Saving model.')
        checkpoint.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=CURRENT_EPOCH,
            total_epochs=TOTAL_EPOCHS,
            output_path=CHECKPOINT_PATH,
        )
