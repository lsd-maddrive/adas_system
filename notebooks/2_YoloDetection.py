#!/usr/bin/env python
# coding: utf-8

# Объединенный датасет [FIX MEне доступен по ссылке](*).
#
# Положить в папку data содержимое так, чтобы были следующие пути:
# * \$(ROOT_DIR)/data/full-rtsd/...
# * \$(ROOT_DIR)/data/full-gt.csv
#
# > *gt_Set_NaN.csv - содержит тот же датасет, но значения колонки Set обнулено*
#
# gt - датафрейм содержащий:
# * имена файлов - поле filename
# * класс знака - поле sign_class
# * координаты знаков
# * в какой набор включен знак - поле Set $\in$ $\{train, valid, test\}$

# In[7]:


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (25, 8)
import numpy as np
import random
import torch
from torch import nn
import pandas as pd
import os
import pathlib
import cv2
import sys

try:
    IN_COLAB = True
    from google.colab import drive

    drive.mount("/content/drive")
except:
    IN_COLAB = False

TEXT_COLOR = "black"

# Зафиксируем состояние случайных чисел
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[2]:


if not IN_COLAB:
    get_ipython().run_line_magic("run", "utils.ipynb")
    PROJECT_ROOT = pathlib.Path(os.path.join(os.curdir, os.pardir))
else:
    PROJECT_ROOT = pathlib.Path("")

DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

if (NOTEBOOKS_DIR / "full-gt.csv").is_file():
    full_gt = pd.read_csv(NOTEBOOKS_DIR / "full-gt.csv")
else:
    full_gt = pd.read_csv(DATA_DIR / "full-gt.csv")

FORMATED_GT_PATH = "formated_full_gt.csv"
FULL_GT_SRC_LEN = len(full_gt.index)
display(full_gt)


# In[3]:


full_gt_unique_filenames = set(full_gt["filename"])
full_gt_unique_filenames_size = len(full_gt_unique_filenames)
get_ipython().run_line_magic("run", "utils.ipynb")
import ast

i = 0

if os.path.isfile(FORMATED_GT_PATH):
    print("FORMATED GT EXIST. LOAD IT")
    formated_full_gt_df = pd.read_csv(FORMATED_GT_PATH, dtype=object)
    # display(formated_full_gt_df)
    formated_full_gt_df["coords"].replace(
        {"\n ": ",", " \s+": " ", "\[ ": "["}, regex=True, inplace=True
    )
    # display(formated_full_gt_df)
    formated_full_gt_df["coords"] = formated_full_gt_df["coords"].apply(
        lambda x: ast.literal_eval(x)
    )

    formated_full_gt_df["size"] = formated_full_gt_df["size"].apply(
        lambda x: ast.literal_eval(x)
    )
else:
    print("FORMATED GT DOESNT EXIST. CREATE IT")
    # get all original filenames
    full_gt_unique_filenames = set(full_gt["filename"])

    formated_full_gt_list = []

    import imagesize

    for src_filename_iterator in list(full_gt_unique_filenames):

        mask = np.in1d(full_gt["filename"], [src_filename_iterator])
        coord_data_arr = full_gt[mask][
            ["x_from", "y_from", "width", "height"]
        ].to_numpy()

        filepath = DATA_DIR / "rtsd-frames" / src_filename_iterator
        origW, origH = imagesize.get(filepath)

        rel_coord = []
        for coord in coord_data_arr:
            # make from x, y, dx, dx -> x1, y1, x2, y2
            CV2RectangleCoords = ConvertAbsTLWH2CV2Rectangle(coord)

            # make from x1, y1, x2, y2 -> x, y, w, h
            CV2CircleCoords = ConvertCV2Rectangle2CenterXYWH(CV2RectangleCoords)

            # make x, y, w, h -> relative x, y, w, h
            rel_instance = MakeRel(CV2CircleCoords, origW, origH)

            rel_coord.append(rel_instance)

        if i % 100 == 0:
            printProgressEnum(i, full_gt_unique_filenames_size)
        i += 1

        formated_full_gt_list.append([str(filepath), rel_coord, [origW, origH]])

    formated_full_gt_df = pd.DataFrame(
        formated_full_gt_list, columns=["filepath", "coords", "size"]
    )
    formated_full_gt_df.to_csv("formated_full_gt.csv", index=False)

formated_full_gt_df


# # simple test

# In[37]:


instance = formated_full_gt_df.iloc[15466]
print(instance)
img = cv2.imread(str(instance["filepath"]))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


h, w = img.shape[0], img.shape[1]
print("Shape:", w, h)


for i in instance["coords"]:

    xywh = UnmakeRel(i, w, h)
    x1y1x2y2 = ConvertCenterXYWH2CV2Rectangle(xywh)
    print("+", MakeRel(x1y1x2y2, w, h))
    print("xywh", xywh)
    print("x1y1x2y2", x1y1x2y2)

    img = cv2.rectangle(
        img, (x1y1x2y2[0], x1y1x2y2[1]), (x1y1x2y2[2], x1y1x2y2[3]), (255, 0, 0), 3
    )

    img = cv2.circle(img, (xywh[0], xywh[1]), xywh[2] // 2, (255, 255, 0), 3)


plt.imshow(img)
plt.show()


# In[5]:


if "set" in formated_full_gt_df.columns:
    print("SET ALREADY EXIST")
else:
    print("SET DOESNT EXIST. LETS CREATE IT")
    formated_full_gt_df_index_count = len(formated_full_gt_df.index)
    TRAIN_SIZE = round(0.7 * formated_full_gt_df_index_count)
    VALID_SIZE = round(0.2 * formated_full_gt_df_index_count)
    TEST_SIZE = round(formated_full_gt_df_index_count - TRAIN_SIZE - VALID_SIZE)

    # print('assert:', TRAIN_SIZE + VALID_SIZE + TEST_SIZE, '==', formated_full_gt_df_index_count)

    assert (
        TRAIN_SIZE + VALID_SIZE + TEST_SIZE == formated_full_gt_df_index_count
    ), "wrong split"
    set_series = (
        pd.Series("test", index=range(TEST_SIZE))
        .append(
            pd.Series("train", index=range(TRAIN_SIZE)).append(
                pd.Series("valid", index=range(VALID_SIZE))
            )
        )
        .sample(frac=1)
        .reset_index(drop=True)
    )
    formated_full_gt_df["set"] = set_series
    display(formated_full_gt_df)
    formated_full_gt_df.to_csv("formated_full_gt.csv", index=False)

display(formated_full_gt_df)


# Now we have pd.DataFrame that contains filenames, list of relative coordinates, corresponding photo resoulutions and marks for set.
#
# Let's make DataSets for it.

# In[98]:


import yaml

YOLO_MODEL_HOME_DIR = DATA_DIR / "YoloV5"
AUGMENT_HOME_DIR = YOLO_MODEL_HOME_DIR / "utils"

if YOLO_MODEL_HOME_DIR not in sys.path:
    sys.path.append(str(YOLO_MODEL_HOME_DIR))

from models.yolo import Model
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from utils.augmentations import (
    Albumentations,
    augment_hsv,
    letterbox,
    random_perspective,
)

hyps_file = YOLO_MODEL_HOME_DIR / "data/hyps" / "hyp.scratch.yaml"
with open(hyps_file, errors="ignore") as f:
    hyp = yaml.safe_load(f)


class CreateDataSet(torch.utils.data.Dataset):
    def __init__(
        self, df, set_label, img_size=640, batch_size=16, augment=False, hyp=None
    ):

        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp

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
                degrees=hyp["degrees"],
                translate=hyp["translate"],
                scale=hyp["scale"],
                shear=hyp["shear"],
                perspective=hyp["perspective"],
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
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
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


# # Simple DataLoader and DataSet wrapper

# In[131]:


def createDataLoaderAndDataSet(
    df, set_label, imgsz, batch_size, hyp=None, augment=False, shuffle=True
):

    from torch.utils.data import DataLoader

    dataset = CreateDataSet(df, set_label, img_size=imgsz, augment=augment)
    batch_size = min(batch_size, len(dataset))

    sampler = None  # distributed.DistributedSampler(dataset, shuffle=shuffle)

    loader = DataLoader(
        dataset,  # InfiniteDataLoader ?
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        # num_workers=nw,  # doesnt work in Windows
        sampler=sampler,
        pin_memory=True,
        collate_fn=CreateDataSet.collate_fn,
    )

    return loader, dataset


IMG_SIZE = 1280
train_loader, train_dataset = createDataLoaderAndDataSet(
    formated_full_gt_df, "test", imgsz=IMG_SIZE, batch_size=20, augment=False
)

img, labels_out, filepath, shapes = train_dataset[1]

imgNT = img.numpy().transpose(1, 2, 0).astype(np.uint8).copy()  # , cv2.COLOR_BGR2RGB)
print(labels_out)
print(filepath)
for coord in labels_out[:, 2:]:
    # print(coord)
    h, w = shapes[0]
    xywh = UnmakeRel(coord, IMG_SIZE, IMG_SIZE)
    x1y1x2y2 = ConvertCenterXYWH2CV2Rectangle(xywh)
    print(x1y1x2y2)
    imgNT = cv2.rectangle(
        imgNT, (x1y1x2y2[0], x1y1x2y2[1]), (x1y1x2y2[2], x1y1x2y2[3]), (255, 0, 0), 3
    )

plt.imshow(imgNT)


# In[118]:


model_cfg_file = YOLO_MODEL_HOME_DIR / "models/yolov5l_custom_anchors.yaml"
model = Model(cfg=model_cfg_file, ch=3, nc=1)


# In[76]:


def train(hyp, opt, device, callbacks):  # path/to/hyp.yaml or hyp dictionary
    (
        save_dir,
        epochs,
        batch_size,
        weights,
        single_cls,
        evolve,
        data,
        cfg,
        resume,
        noval,
        nosave,
        workers,
        freeze,
    ) = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )

    # Directories
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(
        colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items())
    )

    # Save run settings
    if not evolve:
        with open(save_dir / "hyp.yaml", "w") as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / "opt.yaml", "w") as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda = device.type != "cpu"
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = (
        ["item"] if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]
    )  # class names
    assert (
        len(names) == nc
    ), f"{len(names)} names found for nc={nc} dataset in {data}"  # check
    is_coco = isinstance(val_path, str) and val_path.endswith(
        "coco/val2017.txt"
    )  # COCO dataset

    # Model
    check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(
            cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")
        ).to(
            device
        )  # create
        exclude = (
            ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []
        )  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(
            f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}"
        )  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create

    # Freeze
    freeze = [
        f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))
    ]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(
            v.weight, nn.Parameter
        ):  # weight (with decay)
            g1.append(v.weight)

    if opt.optimizer == "Adam":
        optimizer = Adam(
            g0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
        )  # adjust beta1 to momentum
    elif opt.optimizer == "AdamW":
        optimizer = AdamW(
            g0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
        )  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

    optimizer.add_param_group(
        {"params": g1, "weight_decay": hyp["weight_decay"]}
    )  # add g1 with weight_decay
    optimizer.add_param_group({"params": g2})  # add g2 (biases)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
        f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias"
    )
    del g0, g1, g2

    # Scheduler
    if opt.linear_lr:
        lf = (
            lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp["lrf"]) + hyp["lrf"]
        )  # linear
    else:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lf
    )  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # EMA
        if ema and ckpt.get("ema"):
            ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            ema.updates = ckpt["updates"]

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if resume:
            assert (
                start_epoch > 0
            ), f"{weights} training to {epochs} epochs is finished, nothing to resume."
        if epochs < start_epoch:
            LOGGER.info(
                f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs."
            )
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started."
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
    )
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert (
        mlc < nc
    ), f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end")

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = (
        labels_to_class_weights(dataset.labels, nc).to(device) * nc
    )  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(
        round(hyp["warmup_epochs"] * nb), 1000
    )  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)  # init loss class
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\n"
        f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )
    for epoch in range(
        start_epoch, epochs
    ):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = (
                model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            )  # class weights
            iw = labels_to_image_weights(
                dataset.labels, nc=nc, class_weights=cw
            )  # image weights
            dataset.indices = random.choices(
                range(dataset.n), weights=iw, k=dataset.n
            )  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(
            ("\n" + "%10s" * 7)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "labels", "img_size")
        )
        if RANK in [-1, 0]:
            pbar = tqdm(
                pbar, total=nb, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            )  # progress bar
        optimizer.zero_grad()
        for (
            i,
            (imgs, targets, paths, _),
        ) in (
            pbar
        ):  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = (
                imgs.to(device, non_blocking=True).float() / 255
            )  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni,
                        xi,
                        [
                            hyp["warmup_bias_lr"] if j == 2 else 0.0,
                            x["initial_lr"] * lf(epoch),
                        ],
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi, [hyp["warmup_momentum"], hyp["momentum"]]
                        )

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]
                    ]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(
                        imgs, size=ns, mode="bilinear", align_corners=False
                    )

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(device)
                )  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%10s" * 2 + "%10.4g" * 5)
                    % (
                        f"{epoch}/{epochs - 1}",
                        mem,
                        *mloss,
                        targets.shape[0],
                        imgs.shape[-1],
                    )
                )
                callbacks.run(
                    "on_train_batch_end",
                    ni,
                    model,
                    imgs,
                    targets,
                    paths,
                    plots,
                    opt.sync_bn,
                )
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(
                model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"]
            )
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = val.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

            # Update best mAP
            fi = fitness(
                np.array(results).reshape(1, -1)
            )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "wandb_id": loggers.wandb.wandb_run.id if loggers.wandb else None,
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (
                    (epoch > 0)
                    and (opt.save_period > 0)
                    and (epoch % opt.save_period == 0)
                ):
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run(
                    "on_model_save", last, epoch, final_epoch, best_fitness, fi
                )

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(
            f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours."
        )
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = val.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65
                        if is_coco
                        else 0.60,  # best pycocotools results at 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=True,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run(
                            "on_fit_epoch_end",
                            list(mloss) + list(results) + lr,
                            epoch,
                            best_fitness,
                            fi,
                        )

        callbacks.run("on_train_end", last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


# In[29]:


train_dataset[15466]


# In[ ]:


# In[ ]:


# In[ ]:


m.anchors
# 7,12,  10,17,  13,24,  26,20,  17,32,  23,41,  31,54,  41,72,  58,100 # custom anchors


# In[ ]:


t_ = torch.tensor(
    [
        [[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]],  # P3/8-small
        [[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]],  # P4/16-medium
        [[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]],
    ],
    dtype=torch.float16,
)


# In[ ]:


Model


# In[45]:


model.save("asd")
