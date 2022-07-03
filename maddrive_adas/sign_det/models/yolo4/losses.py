import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import math

from addict import Dict


# NOTE - used special iou to calculate not matrix [N, N], but vector with line N
#       Compares bboxes pair by pair
def _bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4xn, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from сxсywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class YoloBboxLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = Dict(config)

        self.strides = config.strides
        self.n_classes = config.n_classes

        self.ciou_ratio = config.bbox_loss.ciou_ratio
        self.match_threshold = config.bbox_loss.gt_match_max_ratio

        self.anchors = np.array(config.anchors).reshape(-1, 2)
        self.anch_masks = config.anchor_masks
        self.outputs_anchors = []

        self.loss_weights = {
            "box": config.bbox_loss.get('bbox_rate', 0.05),
            "obj": config.bbox_loss.get('obj_rate', 1.),
            "cls": config.bbox_loss.get('cls_rate', 0.58)
        }

        for i in range(config.n_outputs):
            _stride = self.strides[i]

            all_anchors_grid = [(w / _stride, h / _stride)
                                for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j]
                                       for j in self.anch_masks[i]], dtype=np.float32)
            self.outputs_anchors.append(masked_anchors)

    def forward(self, y_pred, y_true):
        # Decoding target data from matrix view
        # Shape = [b, n_objects]
        nlabel = (y_true.sum(dim=2) > 0).sum(dim=1)  # number of objects
        batch_ids = torch.cat([torch.full_like(y_true[i, :n_labels, 4], i)
                               for i, n_labels in enumerate(nlabel)])
        batch_ids = batch_ids.type(torch.int64).cpu()
        gt_labels = torch.cat([y_true[i, :n_labels, 4]
                               for i, n_labels in enumerate(nlabel)])
        gt_labels = gt_labels.type(torch.int64).cpu()
        gt_boxes = torch.cat([y_true[i, :n_labels, :4]
                              for i, n_labels in enumerate(nlabel)])

        # xywh -> cxcywh
        gt_boxes[:, :2] = gt_boxes[:, :2] + gt_boxes[:, 2:]/2

        losses = {
            "box": gt_boxes.new_tensor(0),
            "obj": gt_boxes.new_tensor(0),
        }

        if self.n_classes > 1:
            losses['cls'] = gt_boxes.new_tensor(0)

        # NOTE - proposals - it multiplies amount of GTs
        def assign_targets_to_proposals(xy, size, overlap=0.5):
            x, y = xy.T

            ids = [
                torch.arange(len(xy), device=xy.device),
                torch.where((x > 1) & (x % 1 < overlap))[0],  # lt_x
                torch.where((y > 1) & (y % 1 < overlap))[0],  # lt_y
                torch.where((x < size[1] - 1) & (x %
                                                 1 > (1 - overlap)))[0],  # rb_x
                torch.where((y < size[0] - 1) & (y %
                                                 1 > (1 - overlap)))[0],  # rb_y
            ]

            offsets = xy.new_tensor([
                [0, 0],
                [-overlap, 0],
                [0, -overlap],
                [overlap, 0],
                [0, overlap]
            ])
            coordinates = torch.cat([xy[ids[i]] + offsets[i]
                                     for i in range(5)]).long()
            return torch.cat(ids), coordinates

        # TODO - types refactoring!
        for output_id, output in enumerate(y_pred):
            # Output shape: BCHW
            batchsize = output.shape[0]
            out_h = output.shape[2]
            out_w = output.shape[3]
            # xywh + conf + classes
            n_ch = 4 + 1 + self.n_classes

            # Reshape prediction
            # Shape: images (batch), anchors, H, W, [bbox, conf, classes]
            output = output.view(batchsize, -1, n_ch, out_h, out_w)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            _anchors = torch.DoubleTensor(
                self.outputs_anchors[output_id]).to(output.device)
            _stride = self.strides[output_id]

            strided_gt_boxes = gt_boxes / _stride

            # NOTE - https://github.com/Okery/YOLOv5-PyTorch/blob/master/yolo/model/head.py
            def size_matched_idx(wh1, wh2, thresh):
                ratios = wh1[:, None] / wh2[None]
                max_ratios = torch.max(ratios, 1. / ratios).max(2)[0]

                # print(max_ratios)
                return torch.where(max_ratios < thresh)

            def wh_iou(wh1, wh2):
                # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
                wh1 = wh1[:, None]  # [N,1,2]
                wh2 = wh2[None]  # [1,M,2]
                inter = torch.min(wh1, wh2).prod(2)  # [N,M]
                # iou = inter / (area1 + area2 - inter)
                return inter / (wh1.prod(2) + wh2.prod(2) - inter)

            gt_object = torch.zeros_like(output[..., 4])

            # anchor_ious = wh_iou(strided_gt_boxes[:, 2:], _anchors).cpu()
            # IOU_THRESHOLD = 0.2
            # gt_id, anchor_id = np.where(anchor_ious > IOU_THRESHOLD)

            if strided_gt_boxes.shape[0] > 0:
                # print(_anchors)
                # print(strided_gt_boxes)
                gt_id, anchor_id = size_matched_idx(strided_gt_boxes[:, 2:], _anchors, thresh=self.match_threshold)
                # print(gt_id, anchor_id)

                if len(anchor_id) > 0:
                    gt_box_xy = strided_gt_boxes[:, :2][gt_id]
                    ids, grid_xy = assign_targets_to_proposals(
                        gt_box_xy, output.shape[2:4])
                    grid_xy = grid_xy.to(output.device)
                    anchor_id, gt_id = anchor_id[ids], gt_id[ids]

                    batch_id = batch_ids[gt_id].numpy()
                    pred_level = output[batch_id, anchor_id,
                                        grid_xy[:, 1], grid_xy[:, 0]]

                    # pxy = 2 * torch.sigmoid(pred_level[:, :2]) - 0.5 + grid_xy
                    pxy = torch.sigmoid(pred_level[:, :2]) + grid_xy
                    # pwh = 4 * torch.sigmoid(pred_level[:, 2:4]) ** 2 * _anchors[anchor_id]
                    pwh = torch.exp(pred_level[:, 2:4]) * _anchors[anchor_id]
                    pxy = pxy.type(pwh.type())

                    # Predicted
                    pbox = torch.cat((pxy, pwh), dim=1)

                    bxs_iou = _bbox_iou(pbox.T, strided_gt_boxes[gt_id], x1y1x2y2=False, CIoU=True)
                    bxs_iou = bxs_iou.to(gt_object.device)

                    losses["box"] += (1 - bxs_iou).mean()

                    gt_object[batch_id, anchor_id, grid_xy[:, 1], grid_xy[:, 0]] = \
                        self.ciou_ratio * \
                        bxs_iou.clamp(0).type(gt_object.dtype) + \
                        (1 - self.ciou_ratio)

                    if self.n_classes > 1:
                        gt_label = torch.zeros_like(pred_level[..., 5:])
                        gt_label[range(len(gt_id)), gt_labels[gt_id]] = 1

                        losses["cls"] += F.binary_cross_entropy_with_logits(pred_level[..., 5:], gt_label)

            losses["obj"] += F.binary_cross_entropy_with_logits(output[..., 4], gt_object)

        losses = {k: v * self.loss_weights[k] for k, v in losses.items()}
        return sum(losses.values()), losses
