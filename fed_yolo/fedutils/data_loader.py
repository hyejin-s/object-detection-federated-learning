import logging

import copy
import os
import yaml
import numpy as np
from pathlib import Path

import glob
import math
import random
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, distributed
from tqdm import tqdm

from fedmodels.yolov5.utils.augmentations import (
    Albumentations,
    augment_hsv,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from fedmodels.yolov5.utils.general import (
    check_img_size,
    LOGGER,
    NUM_THREADS,
    xyn2xy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from fedmodels.yolov5.utils.torch_utils import torch_distributed_zero_first
from fedmodels.yolov5.utils.dataloaders import (
    create_dataloader,
    verify_image_label,
    InfiniteDataLoader,
    get_hash,
    exif_size,
    exif_transpose,
)

# Parameters
HELP_URL = "https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data"
IMG_FORMATS = [
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
]  # include image suffixes
VID_FORMATS = [
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "wmv",
]  # include video suffixes

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = (
        os.sep + "images" + os.sep,
        os.sep + "labels" + os.sep,
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

def partition_data(data_path, partition, n_nets):
    if os.path.isfile(data_path):
        with open(data_path) as f:
            data = f.readlines()
        n_data = len(data)
    else:
        n_data = len(os.listdir(data_path))
    if partition == "homo":
        total_num = n_data
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    elif partition == "hetero":
        _label_path = copy.deepcopy(data_path)
        label_path = _label_path.replace("images", "labels")
        net_dataidx_map = non_iid_coco(label_path, n_nets)
        # print(net_dataidx_map)

    return net_dataidx_map

def non_iid_coco(label_path, client_num):
    res_bin = {}
    label_path = Path(label_path)
    fs = os.listdir(label_path)
    fs = {i for i in fs if i.endswith(".txt")}
    bin_n = len(fs) // client_num
    # print(f"{len(fs)} files found, {bin_n} files per client")

    id2idx = {}  # coco128
    for i, f in enumerate(fs):
        id2idx[int(f.split(".")[0])] = i

    for b in range(bin_n - 1):
        res = {}
        for f in fs:
            if not f.endswith(".txt"):
                continue

            txt_path = os.path.join(label_path, f)
            txt_f = open(txt_path)
            for line in txt_f.readlines():
                line = line.strip("\n")
                l = line.split(" ")[0]
                if res.get(l) == None:
                    res[l] = set()
                else:
                    res[l].add(f)
            txt_f.close()

        sort_res = sorted(res.items(), key=lambda x: len(x[1]), reverse=True)
        # print(f"{b}th bin: {len(sort_res)} classes")
        # print(res)
        fs = fs - sort_res[0][1]
        # print(f"{len(fs)} files left")

        fs_id = [id2idx[int(i.split(".")[0])] for i in sort_res[0][1]]
        res_bin[b] = np.array(fs_id)

    fs_id = [int(i.split(".")[0]) for i in fs]
    res_bin[b + 1] = np.array(list(fs_id))
    return res_bin

def load_partition_data_coco(args, hyp, model):
    save_dir, epochs, batch_size, weights = (
        Path(args.save_dir),
        args.epochs,
        args.batch_size,
        args.weights,
    )

    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    train_path = data_dict["train"]
    test_path = data_dict["val"]
    train_path = os.path.expanduser(train_path)
    test_path = os.path.expanduser(test_path)

    nc, names = (
        (1, ["item"]) if args.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )  # number classes, names
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in args.img_size_list]

    client_number = args.client_num_in_total
    partition = args.partition_method

    net_dataidx_map = partition_data(
        train_path, partition=partition, n_nets=client_number
    )
    net_dataidx_map_test = partition_data(
        test_path, partition=partition, n_nets=client_number
    )
    train_data_loader_dict = dict()
    test_data_loader_dict = dict()
    train_data_num_dict = dict()
    train_dataset_dict = dict()

    return (
        train_data_num,
        test_data_num,
        train_dataloader_global,
        test_dataloader_global,
        train_data_num_dict,
        train_data_loader_dict,
        test_data_loader_dict,
        nc,
    )


def load_partition_data_custom(args, hyp, model):
    batch_size = args.batch_size

    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    train_path = data_dict["path"] + data_dict["train"]
    test_path = "/hdd/hdd3/coco/images/val2017"
    train_path = os.path.expanduser(train_path)
    test_path = os.path.expanduser(test_path)

    nc, names = (
        (int(data_dict["nc"]), data_dict["names"])
    )  # number classes, names
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = check_img_size(args.img_size, gs, floor=gs * 2)

    client_number = args.client_num_in_total

    train_data_loader_dict = dict()
    test_data_loader_dict = dict()
    train_data_num_dict = dict()
    train_dataset_dict = dict()

    testloader = create_dataloader(
            test_path,
            imgsz,
            batch_size,
            gs,
            hyp=hyp,
            rect=True,
            rank=-1,
            pad=0.5,
            workers=args.worker_num,
        )[0]

    class_list = [51, 60]
    for client_idx in range(args.client_num_in_total):
        # client_idx = int(args.process_id) - 1
        train_path = data_dict["path"] + f"/train_class{class_list[client_idx]}"
        dataloader, dataset = create_dataloader(
            train_path,
            imgsz,
            batch_size,
            gs,
            args,
            hyp=hyp,
            rect=True,
            workers=args.worker_num,
        )

        train_dataset_dict[client_idx] = dataset
        train_data_num_dict[client_idx] = len(dataset)
        train_data_loader_dict[client_idx] = dataloader
        test_data_loader_dict[client_idx] = testloader


    return (
        train_dataset_dict,
        train_data_num_dict,
        train_data_loader_dict,
        test_data_loader_dict,
        testloader,
        nc,
    )
