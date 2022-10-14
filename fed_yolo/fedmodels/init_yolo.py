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
    args.img_size_list.extend(
        [args.img_size_list[-1]] * (2 - len(args.img_size_list))
    )  # extend to 2 sizes (train, test)
    # args.name = "evolve" if args.evolve else args.name
    # args.save_dir = increment_path(
    #     Path(args.project) / args.name, exist_ok=args.exist_ok
    # )  # increment run

    # add checkpoint interval
    # logging.info("add checkpoint interval")
    # args.checkpoint_interval = (
    #     50 if args.checkpoint_interval is None else args.checkpoint_interval
    # )
    # args.server_checkpoint_interval = (
    #     5
    #     if args.server_checkpoint_interval is None
    #     else args.server_checkpoint_interval
    # )

    # Hyperparameters
    with open(args.yolo_hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if "box" not in hyp:
            warn(
                'Compatibility: %s missing "box" which was renamed from "giou" in %s'
                % (args.yolo_hyp, "https://github.com/ultralytics/yolov5/pull/1120")
            )
            hyp["box"] = hyp.pop("giou")

    args.total_batch_size = args.batch_size

    total_batch_size, weights = (args.total_batch_size, args.weights)

    # logging.info(f"Hyperparameters {hyp}")
    # save_dir, epochs, batch_size, total_batch_size, weights = (
    #     Path(args.save_dir),
    #     args.epochs,
    #     args.batch_size,
    #     args.total_batch_size,
    #     args.weights,
    # )

    # # Directories
    # wdir = save_dir / "weights"
    # wdir.mkdir(parents=True, exist_ok=True)  # make dir
    # last = wdir / "last.pt"
    # best = wdir / "best.pt"
    # results_file = save_dir / "results.txt"

    # add file handler
    # logging.info("add file handler")
    # fh = logging.FileHandler(os.path.join(args.save_dir, f"log_{args.process_id}.txt"))
    # fh.setLevel(logging.INFO)
    # logging.getLogger().addHandler(fh)

    # args.last, args.best, args.results_file = last, best, results_file

    # Configure
    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (int(data_dict["nc"]), data_dict["names"])  # number classes, names

    # nc, names = (
    #     (1, ["item"]) if args.single_cls else (int(data_dict["nc"]), data_dict["names"])
    # )  # number classes, names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        args.data,
    )  # check
    args.nc = nc  # change nc to actual number of classes

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
    args.model_stride = model.stride
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [
        check_img_size(x, gs) for x in args.img_size_list
    ]  # verify imgsz are gs-multiples

    hyp["cls"] *= nc / 80.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.class_weights = labels_to_class_weights(train_data_global.dataset.labels, nc).to(
    # device
    # )  # attach class weights
    model.names = names
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(
        round(nbs / total_batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay
    # logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # Save run settings
    # with open(save_dir / "hyp.yaml", "w") as f:
    #     yaml.dump(hyp, f, sort_keys=False)
    # with open(save_dir / "opt.yaml", "w") as f:
    #     # save args as yaml
    #     yaml.dump(args.__dict__, f, sort_keys=False)

    args.hyp = hyp  # add hyperparameters
    args.wandb = wandb

    return model, args
