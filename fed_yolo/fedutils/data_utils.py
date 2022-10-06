import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
from skimage.transform import resize
from timm.data import Mixup
from timm.data import create_transform
# from timm.data.transforms import _pil_interp
from fedmodels.yolov5.utils.general import check_dataset
from fedutils.data_loader import load_partition_data_custom
from torch.utils.data import DataLoader
import yaml

import torch
from torchvision import transforms
import torch.utils.data as data

Image.LOAD_TRUNCATED_IMAGES = True

def create_dataset_and_evalmetrix(args, model):

    if args.dataset == 'coco_custom':

        with open(args.yolo_hyp) as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
            if "box" not in hyp:
                warn(
                    'Compatibility: %s missing "box" which was renamed from "giou" in %s'
                    % (args.yolo_hyp, "https://github.com/ultralytics/yolov5/pull/1120")
                )
                hyp["box"] = hyp.pop("giou")
        
        dataset = load_partition_data_custom(args, hyp, model)
        [ 
            args.train_dataset_dict,
            args.train_data_num_dict,
            args.train_data_loader_dict,
            args.test_data_loader_dict,
            args.test_loader,
            args.num_classes,
        ] = dataset

        args.dis_cvs_files = list(args.train_dataset_dict.keys())
        args.clients_with_len = {name: args.train_data_num_dict[name] for name in args.dis_cvs_files}






