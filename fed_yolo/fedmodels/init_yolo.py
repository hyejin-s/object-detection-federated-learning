import os
import sys

import logging
from pathlib import Path
from warnings import warn
import yaml
import torch

# from data.data_loader import load_partition_data_coco
from .yolov5.utils.general import (
    increment_path,
    check_file,
    check_img_size,
)

from .yolov5.utils.general import intersect_dicts
from .yolov5.models.yolo import Model as YOLOv5

try:
    import wandb
except ImportError:
    wandb = None
    logging.info(
        "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)"
    )

def init_yolo(args, device="cpu"):
    # init settings
    args.yolo_hyp = args.yolo_hyp or (
        "hyp.finetune.yaml" if args.weights else "hyp.scratch.yaml"
    )
    args.data_conf, args.yolo_cfg, args.yolo_hyp = (
        check_file(args.data_conf),
        check_file(args.yolo_cfg),
        check_file(args.yolo_hyp),
    )  # check files
    assert len(args.yolo_cfg) or len(
        args.weights
    ), "either yolo_cfg or weights must be specified"

    # Hyperparameters
    with open(args.yolo_hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if "box" not in hyp:
            warn(
                'Compatibility: %s missing "box" which was renamed from "giou" in %s'
                % (args.yolo_hyp, "https://github.com/ultralytics/yolov5/pull/1120")
            )
            hyp["box"] = hyp.pop("giou")

    total_batch_size, weights = (args.batch_size, args.weights)

    # Configure
    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    nc, names = (int(data_dict["nc"]), data_dict["names"])  # number classes, names

    # Model
    print("weights:", weights)
    # if args.model.lower() == "yolov5":
    pretrained = weights.endswith(".pt")
    if pretrained:
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        if hyp.get("anchors"):
            ckpt["model"].yaml["anchors"] = round(hyp["anchors"])  # force autoanchor
        model = YOLOv5(args.yolo_cfg or ckpt["model"].yaml, ch=3, nc=nc).to(
            device
        )  # create
        exclude = (
            ["anchor"] if args.yolo_cfg or hyp.get("anchors") else []
        )  # exclude keys
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = intersect_dicts(
            state_dict, model.state_dict(), exclude=exclude
        )  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logging.info(
            "Transferred %g/%g items from %s"
            % (len(state_dict), len(model.state_dict()), weights)
        )  # report
    else:
        model = YOLOv5(args.yolo_cfg, ch=3, nc=nc).to(device)  # create

    # print(model)
    hyp["cls"] *= nc / 80.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.names = names
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(
        round(nbs / total_batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay

    return model, hyp
