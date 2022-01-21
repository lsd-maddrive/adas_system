from utils.augmentations import (
    Albumentations,
    augment_hsv,
    letterbox,
    random_perspective,
)
from utils.general import non_max_suppression, scale_coords

import torch
import cv2
import numpy as np
from torch import nn
import random
import pandas as pd
import matplotlib.pyplot as plt

from models.experimental import Ensemble
from models.yolo import Detect, Model
from models.common import Conv


def printProgressEnum(index, length, label=None):
    print(
        "\r{}Progress: {}/{} ({:.2f}%)".format(
            label if label != None else "",
            index + 1,
            length,
            100 * (index + 1) / length,
        ),
        flush=True,
        end="",
    )


def showTensorPicture(tensor_image, label=None):
    # img = tensor_image.permute(1, 2, 0)
    img = cv2.cvtColor(tensor_image.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    if label:
        plt.title(label)
    plt.show()


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


class CreateDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        set_label,
        hyp_arg,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
    ):

        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp_arg

        self.df = df[df["set"] == set_label]
        self.albumentations = Albumentations() if augment else None

    def loadImage(self, instance):
        path, (w0, h0) = instance["filepath"], instance["size"]
        im = cv2.imread(path)
        assert im is not None, f"Image Not Found {path}"

        r = self.img_size / max(h0, w0)  # ratio

        if r != 1:  # if sizes are not equal
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA
                if r < 1 and not self.augment
                else cv2.INTER_LINEAR,
            )
        return im, (h0, w0), im.shape[:2]

    def __getitem__(self, index):

        # locate img info from DataFrame
        instance = self.df.iloc[index]

        # get Img, src height, width and resized height, width
        img, (h0, w0), (h, w) = self.loadImage(instance)

        shape = self.img_size

        # make img square
        # print('>', (img>1).sum())
        # print('<=', (img<=1).sum())
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        # print(pad)
        # store core shape info
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # add class to labels. We have 1 class, so just add zeros into first column
        labels = np.array(instance["coords"])
        labels = np.c_[np.zeros(labels.shape[0]), labels]
        # print(labels)

        # fix labels location caused by letterbox
        labels[:, 1:] = xywhn2xyxy(
            labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
        )

        if self.augment:
            img, labels = random_perspective(
                img,
                labels,
                degrees=self.hyp["degrees"],
                translate=self.hyp["translate"],
                scale=self.hyp["scale"],
                shear=self.hyp["shear"],
                perspective=self.hyp["perspective"],
            )

        labels[:, 1:5] = xyxy2xywhn(
            labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=False, eps=1e-3
        )

        # YOLO augmentation technique (!copy-paste!)
        if self.augment:
            # print('augm for', index, instance['filepath'])
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(
                img,
                hgain=self.hyp["hsv_h"],
                sgain=self.hyp["hsv_s"],
                vgain=self.hyp["hsv_v"],
            )

            # Flip up-down
            if random.random() < self.hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < self.hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        nl = len(labels)

        # why out size (?, 6)??
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, instance["filepath"], shapes

    def __len__(self):
        return len(self.df.index)

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def createDataLoaderAndDataSet(
    df,
    set_label,
    imgsz,
    hyp_arg,
    batch_size,
    hyp=None,
    augment=False,
    shuffle=True,
    nw=0,
):

    from torch.utils.data import DataLoader

    dataset = CreateDataSet(df, set_label, hyp_arg, img_size=imgsz, augment=augment)
    batch_size = min(batch_size, len(dataset))

    sampler = None  # distributed.DistributedSampler(dataset, shuffle=shuffle)

    loader = DataLoader(
        dataset,  # InfiniteDataLoader ?
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,  # doesnt work in Windows
        sampler=sampler,
        pin_memory=True,
        collate_fn=CreateDataSet.collate_fn,
    )

    return loader, dataset


class makeDetectFromModel(nn.Module):
    def __init__(self, model, device=None):
        super().__init__()

        ensemble = Ensemble()
        ensemble.append(model).float().eval()

        for m in model.modules():
            if type(m) in [
                nn.Hardswish,
                nn.LeakyReLU,
                nn.ReLU,
                nn.ReLU6,
                nn.SiLU,
                Detect,
                Model,
            ]:
                m.inplace = True  # pytorch 1.7.0 compatibility
                if type(m) is Detect:
                    if not isinstance(
                        m.anchor_grid, list
                    ):  # new Detect Layer compatibility
                        delattr(m, "anchor_grid")
                        setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        self.model = model
        self.device = device

    def forward(self, im):
        return self.model(im)[0]

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        if (
            isinstance(self.device, torch.device) and self.device.type != "cpu"
        ):  # only warmup GPU models
            im = (
                torch.zeros(*imgsz)
                .to(self.device)
                .type(torch.half if half else torch.float)
            )  # input image
            self.forward(im)  # warmup

    @staticmethod
    def translatePreds(
        pred,
        nn_img_size,
        source_img_size,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
    ):

        pred = non_max_suppression(
            pred,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic,
            multi_label=multi_label,
            labels=labels,
            max_det=max_det,
        )

        ret_dict = {
            "coords": [],
            "relative_coords": [],
            "class": [],
            "confs": [],
            "count": 0,
        }

        for i, det in enumerate(pred):
            if len(det):
                ret_dict["relative_coords"].append(det[:, :4])
                det[:, :4] = scale_coords(
                    nn_img_size, det[:, :4], source_img_size
                ).round()

                for *xyxy, conf, cls in reversed(det):
                    ret_dict["coords"].append(list(map(int, xyxy)))
                    ret_dict["confs"].append(float(conf))
                    ret_dict["class"].append(int(cls))

                    ret_dict["count"] += 1

        return ret_dict


def UnmakeRel(coords, w, h):
    return list(map(int, [coords[0] * w, coords[1] * h, coords[2] * w, coords[3] * h]))


def MakeRel(coords, w, h):
    return list(
        map(float, [coords[0] / w, coords[1] / h, coords[2] / w, coords[3] / h])
    )


def ConvertAbsTLWH2CV2Rectangle(coords):
    return list(
        map(int, [coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]])
    )


def ConvertCenterXYWH2CV2Rectangle(coords):
    return list(
        map(
            int,
            [
                coords[0] - coords[2] / 2,
                coords[1] - coords[3] / 2,
                coords[0] + coords[2] / 2,
                coords[1] + coords[3] / 2,
            ],
        )
    )


def ConvertCV2Rectangle2CenterXYWH(coords):
    return list(
        map(
            int,
            [
                (coords[2] + coords[0]) / 2,
                (coords[3] + coords[1]) / 2,
                coords[2] - coords[0],
                coords[3] - coords[1],
            ],
        )
    )


def getRandomFromDataset(gt: pd.DataFrame, label="test"):
    gt = gt[gt["is_present"] == 1]
    random_instance = gt[gt["set"] == label].sample(1)
    img_path = DATA_DIR / "merged-rtsd" / random_instance["filename"].values[0]
    sign_class = random_instance["sign_class"].values[0]
    # print(img_path)
    img = cv2.imread(str(img_path))
    # print(img)
    img_NT = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (160, 160))
    img_T = torch.Tensor.permute(torch.Tensor(img), [2, 0, 1]).div(255)

    return img_NT, img_T.cuda(), sign_class, img_path


def getMaxIndex(t_arr):
    t_arr = t_arr.cpu().detach().numpy()
    return np.argmax(t_arr)


def getLabelFromModelOutput(t_arr, MODEL_CLASS_UNMAP_):
    max_arg = getMaxIndex(t_arr)
    print(max_arg)
    print(MODEL_CLASS_UNMAP_)
    return MODEL_CLASS_UNMAP_[max_arg]


def showImg(img, label=None):
    plt.imshow(img)
    if label:
        plt.title(label)
    plt.show()
