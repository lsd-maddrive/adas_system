from typing import Any, Tuple
import numpy as np
import torch


def boxes_xyxy_2_mask(
    boxes,
    mask_shape: Tuple[int, int],
    three_channel: bool = True,
):
    mask = np.zeros(mask_shape, dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 1

    if three_channel:
        mask = mask[..., None]

    return mask


def xyxy_2_xywh(boxes):
    if boxes.shape[0] == 0:
        return boxes

    boxes = boxes.copy()
    boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, :2]
    return boxes


def xywh_2_xyxy(boxes):
    if boxes.shape[0] == 0:
        return boxes

    boxes = boxes.copy()
    boxes[:, 2:4] = boxes[:, 2:4] + boxes[:, :2]
    return boxes


def clip_xywh_bboxes(boxes, src_shape: tuple):
    if boxes.size == 0:
        return boxes

    boxes = xywh_2_xyxy(boxes)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src_shape[0])

    return xyxy_2_xywh(boxes)


def merge_boxes_n_labels(
    boxes, labels
):
    if labels.shape[0] == 0:
        return np.ndarray((0, 5))

    result = np.hstack((boxes, labels[:, None]))
    return result


def intersection_xyxy_old(boxes_a, boxes_b):
    xy2_box_a = np.expand_dims(boxes_a[:, 2:4], 1)
    xy2_box_b = np.expand_dims(boxes_b[:, 2:4], 0)
    xy1_box_a = np.expand_dims(boxes_a[:, :2], 1)
    xy1_box_b = np.expand_dims(boxes_b[:, :2], 0)

    max_xy = np.minimum(xy2_box_a, xy2_box_b)
    min_xy = np.maximum(xy1_box_a, xy1_box_b)
    return np.clip(max_xy - min_xy, 0, None).prod(axis=2)  # inter


def intersection_xyxy(boxes_a, boxes_b):
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    br = np.minimum(boxes_a[:, None, 2:4], boxes_b[:, 2:4])
    return np.clip(br - tl, 0, None).prod(axis=2)  # inter


def intersection_xywh(boxes_a, boxes_b):
    a_br = boxes_a[:, None, :2] + boxes_a[:, None, 2:]
    b_br = boxes_b[:, :2] + boxes_b[:, 2:]

    tl = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    br = np.minimum(a_br, b_br)

    en = (tl < br).astype(tl.dtype).prod(axis=2)
    area_i = np.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i


def iou_xywh_numpy(boxes_a, boxes_b):
    assert boxes_a.shape[1] == 4, "boxes must have shape [:, 4]"
    assert boxes_b.shape[1] == 4, "boxes must have shape [:, 4]"

    area_a = np.prod(boxes_a[:, 2:], axis=1)
    area_b = np.prod(boxes_b[:, 2:], axis=1)

    area_i = intersection_xywh(boxes_a, boxes_b)
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    return iou


def iou_xyxy_numpy(boxes_a, boxes_b):
    assert boxes_a.shape[1] == 4, "boxes must have shape [:, 4]"
    assert boxes_b.shape[1] == 4, "boxes must have shape [:, 4]"

    inter = intersection_xyxy(boxes_a, boxes_b)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b - inter
    out = inter / union
    return out


def iou_wh_numpy(boxes_a, boxes_b):
    assert boxes_a.shape[1] == 2, "boxes must have shape [:, 2]"
    assert boxes_b.shape[1] == 2, "boxes must have shape [:, 2]"

    # intersection
    br = np.minimum(boxes_a[:, None], boxes_b)

    area_a = np.prod(boxes_a, axis=1)
    area_b = np.prod(boxes_b, axis=1)

    area_i = np.prod(br, 2)
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    return iou


def diou_xywh_torch(boxes_a, boxes_b):
    assert boxes_a.shape[1] == 4
    assert boxes_b.shape[1] == 4

    a_tl = boxes_a[:, None, :2]
    b_tl = boxes_b[:, :2]

    a_sz = boxes_a[:, 2:]
    b_sz = boxes_b[:, 2:]

    a_br = a_tl + boxes_a[:, None, 2:]
    b_br = b_tl + b_sz

    # intersection top left
    tl = torch.max(a_tl, b_tl)
    # intersection bottom right
    br = torch.min(a_br, b_br)
    # convex (smallest enclosing box) top left and bottom right
    con_tl = torch.min(a_tl, b_tl)
    con_br = torch.max(a_br, b_br)

    area_a = torch.prod(a_sz, 1)
    area_b = torch.prod(b_sz, 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    # centerpoint distance squared
    a_cntr = (a_tl + a_br) / 2
    b_cntr = (b_tl + b_br) / 2
    rho2 = ((a_cntr - b_cntr) ** 2).sum(dim=-1)

    c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
    return iou - rho2 / c2


class TorchIoUBBox(object):
    def __init__(self, method="vanila", format_="xywh"):
        self.method = method.lower()

        heads = {
            "xywh": self._xywh_head,
            "xyxy": self._xyxy_head,
            "xcycwh": self._xcycwh_head,
        }

        self._head = heads[format_]

    def __call__(self, boxes_a, boxes_b):
        return self.boxes_iou(
            boxes_a,
            boxes_b,
        )

    def _xyxy_head(self, boxes_a, boxes_b):
        a_tl = boxes_a[:, None, :2]
        b_tl = boxes_b[:, :2]

        a_br = boxes_a[:, None, 2:]
        b_br = boxes_b[:, 2:]

        a_sz = boxes_a[:, 2:] - boxes_a[:, :2]
        b_sz = b_br - b_tl

        return (a_tl, b_tl, a_br, b_br, a_sz, b_sz)

    def _xywh_head(self, boxes_a, boxes_b):
        a_tl = boxes_a[:, None, :2]
        b_tl = boxes_b[:, :2]

        a_sz = boxes_a[:, 2:]
        b_sz = boxes_b[:, 2:]

        a_br = a_tl + boxes_a[:, None, 2:]
        b_br = b_tl + b_sz

        return (a_tl, b_tl, a_br, b_br, a_sz, b_sz)

    def _xcycwh_head(self, boxes_a, boxes_b):
        a_tl = boxes_a[:, None, :2] - boxes_a[:, None, 2:] / 2
        b_tl = boxes_b[:, :2] - boxes_b[:, 2:]

        a_sz = boxes_a[:, 2:]
        b_sz = boxes_b[:, 2:]

        a_br = a_tl + boxes_a[:, None, 2:]
        b_br = b_tl + b_sz

        return (a_tl, b_tl, a_br, b_br, a_sz, b_sz)

    def boxes_iou(self, boxes_a, boxes_b, GIoU=False, DIoU=False, CIoU=False):
        assert boxes_a.shape[1] == 4
        assert boxes_b.shape[1] == 4

        a_tl, b_tl, a_br, b_br, a_sz, b_sz = self._head(boxes_a, boxes_b)

        # intersection top left
        tl = torch.max(a_tl, b_tl)
        # intersection bottom right
        br = torch.min(a_br, b_br)
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(a_tl, b_tl)
        con_br = torch.max(a_br, b_br)

        area_a = torch.prod(a_sz, 1)
        area_b = torch.prod(b_sz, 1)

        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
        area_u = area_a[:, None] + area_b - area_i
        iou = area_i / area_u

        def giou_tail():
            area_c = torch.prod(con_br - con_tl, 2)
            return iou - (area_c - area_u) / area_c

        def diou_tail():
            # centerpoint distance squared
            a_cntr = (a_tl + a_br) / 2
            b_cntr = (b_tl + b_br) / 2
            rho2 = ((a_cntr - b_cntr) ** 2).sum(dim=-1)

            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            return iou - rho2 / c2

        def ciou_tail():
            w1 = a_sz[:, 0]
            h1 = a_sz[:, 1]
            w2 = b_sz[:, 0]
            h2 = b_sz[:, 1]

            # centerpoint distance squared
            a_cntr = (a_tl + a_br) / 2
            b_cntr = (b_tl + b_br) / 2
            rho2 = ((a_cntr - b_cntr) ** 2).sum(dim=-1)

            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            v = (4 / math.pi ** 2) * torch.pow(
                torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2
            )
            with torch.no_grad():
                alpha = v / (1 - iou + v)
            return iou - (rho2 / c2 + v * alpha)

        def vanila_tail():
            return iou

        tails = {
            "vanilla": vanila_tail,
            "giou": giou_tail,
            "diou": diou_tail,
            "ciou": ciou_tail,
        }

        return tails[self.method]()
    