from typing import List

import cv2
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from maddrive_adas.models.yolo import Model
from maddrive_adas.utils.metrics import ap_per_class
from maddrive_adas.utils.general import (
    box_iou, non_max_suppression,
    scale_coords, xywh2xyxy,
)


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    # IoU above threshold and classes match
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu(
        ).detach().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def write_valid_images(
    writer: SummaryWriter,
    images: np.ndarray,
    actual: List[torch.Tensor],
    prediction: List[torch.Tensor],
    epoch,
    actual_color=(255, 255, 0),
    pred_color=(255, 0, 255),
):
    resulting_images = []
    for im, actual_boxes, pred_boxes in zip(images, actual, prediction):
        im = (im * 255).astype(np.uint8).copy()
        for box_coords in actual_boxes:
            im = cv2.rectangle(
                im,
                (box_coords[0] - box_coords[2] // 2, box_coords[1] - box_coords[3] // 2),
                (box_coords[0] + box_coords[2] // 2, box_coords[1] + box_coords[3] // 2),
                color=actual_color,
                thickness=2
            )

        for box_coords in pred_boxes:
            box_coords = box_coords.cpu().numpy().astype('int')
            im = cv2.rectangle(
                im,
                (box_coords[0], box_coords[1]),
                (box_coords[2], box_coords[3]),
                color=pred_color,
                thickness=2
            )
        resulting_images.append(im)

    writer.add_images(
        'Validation',
        img_tensor=np.array(resulting_images),
        global_step=epoch,
        dataformats='NHWC'
    )


@torch.no_grad()
def valid_epoch(
        model: Model,
        dataloader: DataLoader,
        epoch,
        iou_thres=0.5,
        conf_thres=0.01,
        max_det=50,
        half=True,
        writer: SummaryWriter = None,
):
    model.eval()

    nc = 1
    single_cls = True

    # get model device, PyTorch model
    device, *_ = next(model.parameters()).device, True, False, False

    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    names = {k: v for k, v in enumerate(
        model.names if hasattr(model, 'names') else model.module.names)}

    # s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, total=len(dataloader),  # desc=s,
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=False)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):

        im = im.to(device)  # , non_blocking=True)
        targets: torch.Tensor = targets.to(device)

        im = im.half() if half else im.float()
        im /= 255
        nb, _, height, width = im.shape

        # Inference
        out, _ = model(im)

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = []  # for autolabelling
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb,
                                  multi_label=False, agnostic=single_cls, max_det=max_det)

        # fix targets
        prev_idx = -1
        targets_per_image: List[List] = []
        for y in targets.cpu().numpy():
            current_idx = int(y[0])
            if current_idx != prev_idx:
                prev_idx = current_idx
                targets_per_image.append([])
            targets_per_image[-1].append(y.astype('int')[2:])

        write_valid_images(
            writer=writer,
            images=im.cpu().numpy().transpose(0, 2, 3, 1),
            actual=targets_per_image,
            prediction=out,
            epoch=epoch
        )
        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            shape = shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                 torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        torch.cuda.empty_cache()

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir='.', names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    return mp, mr, map50, map, maps
