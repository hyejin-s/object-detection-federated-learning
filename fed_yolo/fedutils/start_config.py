import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as torch_models

# from efficientnet_pytorch import EfficientNet
from fedmodels import build_model

from fedmodels.init_yolo import init_yolo
from torch.nn import Linear
from .config_swin import get_config


def print_options(args, model):
    message = ""

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += (
        "================ FL train of %s with total model parameters: %2.1fM  ================\n"
        % (args.FL_platform, num_params)
    )

    message += "++++++++++++++++ Other Train related parameters ++++++++++++++++ \n"

    for k, v in sorted(vars(args).items()):
        comment = ""
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "++++++++++++++++  End of show parameters ++++++++++++++++ "

    ## save to disk of current log

    args.file_name = os.path.join(args.output_dir, "log_file.txt")

    with open(args.file_name, "wt") as args_file:
        args_file.write(message)
        args_file.write("\n")

    print(message)


def init_configure(args, vis=False):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not args.device == "cpu":
        torch.cuda.manual_seed(args.seed)

    if "YOLOv5" in args.FL_platform:
        print("We use YOLOv5")
        # Model
        model, args = init_yolo(args=args, device=args.device)

    # set output parameters
    print(args.optimizer_type)
    args.name = (
        args.net_name
        + "_lr_"
        + str(args.learning_rate)
        + "_Pretrained_"
        + str(args.Pretrained)
        + "_optimizer_"
        + str(args.optimizer_type)
        + "_WUP_"
        + str(args.warmup_steps)
        + "_Round_"
        + str(args.max_communication_rounds)
        + "_local_epochs_"
        + str(args.local_epoch)
        + "_Seed_"
        + str(args.seed)
    )

    args.output_dir = os.path.join("output", args.FL_platform, args.dataset, args.name)
    os.makedirs(args.output_dir, exist_ok=True)

    print_options(args, model)

    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}

    return model
