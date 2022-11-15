# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import math
from pickle import TRUE
import warnings
import argparse
import random
import numpy as np
from copy import deepcopy
import yaml
from tqdm import tqdm

import torch
from torch.optim import lr_scheduler
import torch.nn as nn

from pathlib import Path
from fedmodels.yolov5.utils.loss import ComputeLoss
from fedmodels.yolov5.utils.torch_utils import smart_optimizer
from fedmodels.yolov5.utils.general import check_img_size, check_file, check_yaml, increment_path
from fedmodels.yolov5.utils.callbacks import Callbacks
import fedmodels.yolov5.val as validate
from fedmodels.yolov5.train import train

from fedutils.data_loader import load_partition_data_custom, load_server_data
from fedutils.util import partial_client_selection, average_model
from fedutils.start_config import init_configure
from fedutils.scheduler import setup_scheduler


import wandb
wandb.login()


def main(args):
    args.data, args.cfg, args.hyp, args.weights, args.project = \
        check_file(args.data), check_yaml(args.cfg), check_yaml(args.hyp), str(args.weights), str(args.project)  # checks
    assert len(args.cfg) or len(args.weights), 'either --cfg or --weights must be specified'
    if args.evolve:
        if args.project == str('runs/train'):  # if default project name, rename to runs/evolve
            args.project = str('runs/evolve')
        args.exist_ok, args.resume = args.resume, False  # pass resume to exist_ok and disable resume
    if args.name == 'cfg':
        args.name = Path(args.cfg).stem  # use model.yaml as name
    args.save_dir = str(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))
    
    # server participation
    if args.central:
        print(" ========== train with server (central dataset) ========== ")
        name = (
            args.net_name
            # + "_lr_"
            # + str(args.hyp["lr0"])
            + "_round_"
            + str(args.rounds)
            + "_server_epochs_"
            + str(args.server_local_epoch)
            + "_local_epochs_"
            + str(args.epochs)
            + "_optimizer_"
            + str(args.optimizer_type)
            + "_weights_"
            + str(args.ratio_weights)
            + "_seed_"
            + str(args.seed)
        )
    else:
        print(" ========== train only nodes ========== ")
        name = (
            args.net_name
            # + "_lr_"
            # + str(args.hyp["lr0"])
            + "_round_"
            + str(args.rounds)
            + "_local_epochs_"
            + str(args.epochs)
            + "_optimizer_"
            + str(args.optimizer_type)
            + "_weights_"
            + str(args.ratio_weights)
            + "_seed_"
            + str(args.seed)
        )
    
    num = 0
    exp_dir = os.path.join(args.save_dir, f"{name}_{num}")
    while os.path.exists(exp_dir):
        num += 1
        exp_dir = os.path.join(args.save_dir, f"{name}_{num}")
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # if args.wandb:
    wandb.init(
        project=args.project,
        name=f"{name}_{num}",
        config={
            # "learning_rate": args.hyp["lr0"],
            "epoch": args.rounds,
            "server_epoch": args.server_local_epoch,
            "local_epoch": args.epochs,
            "batch_size": args.batch_size
        })
        
    # initialization    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, hyp = init_configure(args, device, exp_dir) # default: hyp.scratch-low

    # training, validating, and testing
    ''' prepare dataset '''
    with open(args.data) as f: # default: coco_fl
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    shuffle = args.shuffle
    batch_size, imgsz = args.batch_size, args.imgsz
    clients, num_clients = args.clients, len(args.clients)
    dataset = load_partition_data_custom(args, hyp, model, data_dict, batch_size, imgsz, clients, shuffle)
    [
        train_dataset_dict,
        train_data_num_dict,
        _,
        test_loader
    ] = dataset

    server_loader, server_dataset = load_server_data(args, hyp, model, data_dict, batch_size, imgsz, shuffle)

    # all the clients joined in the train
    clients_keys = list(train_dataset_dict.keys()) 
    clients_data_num = {
        name: train_data_num_dict[name] for name in clients_keys
    }

    model.to(device)
    compute_loss = ComputeLoss(model)

    results_name = ["P", "R", "mAP@.5", "mAP@.5-.95", "val_loss(box)", "val_loss(obj)", "val_loss(cls)"]
    # checking initial model performace
    server_results, server_maps, _ = validate.run(
        data_dict,
        weights=args.weights,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        half=True,
        model=model,
        dataloader=test_loader,
        plots=False,
        verbose=False,
        compute_loss=compute_loss,
    )
    
    # log
    wandb.log({f"SERVER_RESULTS_{results_name[3]}": server_results[3]}, step=0)
    for i in range(len(args.clients)):
        wandb.log({f"SERVER_CLIENTS_{i}_CLASS_{args.clients[i]}": server_maps[args.clients[i]]}, step=0)
            
    # Configuration for FedAVG, prepare model, optimizer, and scheduler
    model_all, _, _ = partial_client_selection(args, model, hyp)
    model_server = deepcopy(model).cpu() # server for update
    model_avg = deepcopy(model).cpu() # average server
    
    ##############################
    ##      START TRAINING      ##  
    ##############################
    print("=============== start training ===============")
    for round in range(args.rounds):
        
        if args.ratio_weights == "data_num":
        # Get the quantity of clients joined in the FL train for updating the clients weights
            curr_total_client_lens = 0
            for client in clients_keys:
                curr_total_client_lens += clients_data_num[client]
            
        # if server participate the training
        if args.central:
            # the ratio of clients for updating the clients weights
            if args.ratio_weights == "data_num":
                curr_total_client_lens += len(server_dataset)
                server_weight = len(server_dataset) / curr_total_client_lens
            elif args.ratio_weights == "same":
                server_weight = 1 / (len(args.clients)+1)

            print(
                "train the server", "of communication round", round
            )
            
            # sever's train dataset 
            server_data_path = data_dict["path"] + "server"
            model_server = model_server.to(device)
            model_server, server_results, server_maps = train(hyp, args, model_server, server_data_path, device)    

            # log
            wandb.log({f"SERVER_CENTRAL_RESULTS_{results_name[3]}": server_results[3]})
            for i in range(len(args.clients)):
                wandb.log({f"SERVER_CENTRAL_RESULTS_CLASS_{args.clients[i]}": server_maps[args.clients[i]]})


            print(" ========== finish to train server ========== ")           
    
        clients_weights = {}                            
        for proxy_single_client in range(len(args.clients)):
            print(
                "train the client", proxy_single_client, "of communication round", round
            )

            # the ratio of clients for updating the clients weights
            if args.ratio_weights == "data_num":
                clients_weights[proxy_single_client] = (
                clients_data_num[proxy_single_client] / curr_total_client_lens
            )
            elif args.ratio_weights == "same":
                clients_weights[proxy_single_client] = 1 / (len(args.clients)+1)
            
            model = model_all[proxy_single_client].to(device)

            # client's train dataset 
            data_path = data_dict["path"] + f"/node_{proxy_single_client+1}_class_{clients[proxy_single_client]}"
            model, results, maps = train(hyp, args, model, data_path, device)    
            model_all[proxy_single_client] = deepcopy(model)
            
            # log
            wandb.log({f"CLIENTS_{proxy_single_client}_RESULTS_{results_name[3]}": results[3]})
            for i in range(len(args.clients)):
                wandb.log({f"CLIENTS_{proxy_single_client}_CLASS_{args.clients[i]}": maps[args.clients[i]]})
            
        """ ---- model average and eval ---- """

        if args.central:
            average_model(args, device, model_avg, model_all, model_server, server_weight, clients_weights)
        else:
            weight=None
            average_model(args, device, model_avg, model_all, model_server, weight, clients_weights)
        
        # then evaluate server
        model_avg.to(device)
        compute_loss = ComputeLoss(model_avg)
        server_results, server_maps, _ = validate.run(
            data_dict,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            half=True,
            model=model_avg,
            single_cls=False,
            dataloader=test_loader,
            plots=False,
            compute_loss=compute_loss,
        )
        
        wandb.log({f"SERVER_RESULTS_{results_name[3]}": server_results[3]})
        for i in range(len(args.clients)):
            wandb.log({f"SERVER_CLIENTS_{i}_CLASS_{args.clients[i]}": server_maps[args.clients[i]]})

    print("================ end training ================ ")
    ## model save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General DL parameters
    parser.add_argument(
        "--net_name",
        type=str,
        default="yolov5s",
        help="Basic Name of this run with detailed network-architecture selection. ",
    )

    parser.add_argument(
        "--dataset",
        choices=["all", "per_class"],
        default="per_class",
        help="Node; Which dataset.",
    )
    
    parser.add_argument(
        "--optimizer_type",
        default="SGD",
        choices=["SGD", "ADAMW"],
        type=str,
        help="Ways for optimization.",
    )
    # parser.add_argument("--num_workers", default=8, type=int, help="num_workers")
    # parser.add_argument(
    #     "--weight_decay",
    #     default=0,
    #     choices=[0.05, 0],
    #     type=float,
    #     help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper",
    # )
    # parser.add_argument(
    #     "--grad_clip",
    #     action="store_true",
    #     default=True,
    #     help="whether gradient clip to 1 or not",
    # )
    # parser.add_argument(
    #     "--img_size", default=640, type=int, help="Final train resolution"
    # )
    # parser.add_argument(
    #     "--batch_size", default=256, type=int, help="Local batch size for training."
    # )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )  # 99999

    ## section 2:  DL learning rate related
    # parser.add_argument(
    #     "--decay_type",
    #     choices=["cosine", "linear", "step"],
    #     default="step",
    #     help="How to decay the learning rate.",
    # )
    # parser.add_argument(
    #     "--warmup_steps",
    #     default=100,
    #     type=int,
    #     help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.",
    # )
    # parser.add_argument(
    #     "--step_size",
    #     default=30,
    #     type=int,
    #     help="Period of learning rate decay for step size learning rate decay",
    # )
    # parser.add_argument(
    #     "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    # )
    # parser.add_argument(
    #     "--learning_rate",
    #     default=1e-6,
    #     type=float,
    #     help="The initial learning rate for SGD.",
    # )

    ## FL related parameters
    # parser.add_argument(
    #     "--local_epoch", default=1, type=int, help="Local training epoch in FL"
    # )
    parser.add_argument(
        "--server_local_epoch", default=1, type=int, help="Local training epoch in FL"
    )
    parser.add_argument(
        "--rounds",
        default=50,
        type=int,
        help="Total communication rounds",
    )
    parser.add_argument(
        "--clients", nargs="+", default=[56, 60], help="56: chair, 60: table"
    )

    parser.add_argument(
        "--central",
        default=False,
        help="whether server participate training with server dataset",
    )

    parser.add_argument(
        "--grad_surgery",
        default=False,
        help="whether applying gradient surgery",
    )

    parser.add_argument("--shuffle", default=True, help="dataset shuffle.")
    parser.add_argument(
        "--ratio_weights",
        type=str,
        default="data_num",
        choices=["same", "data_num"],
        help="ratio of weights among node and clients"
    )

    # saving ------------------
    parser.add_argument("--save_dir", default="./exp/")
    parser.add_argument("--wandb", default=True)
    parser.add_argument("--project", default="FedYOLO")
    
    # yolo hyper-parameters
    parser.add_argument('--weights', type=str, default="/hdd/hdd3/coco_fl/best.pt", help='initial weights path')
    parser.add_argument('--cfg', type=str, default="fedmodels/yolov5/models/yolov5s.yaml", help='model.yaml path')
    parser.add_argument('--data', type=str, default="data/coco_custom.yaml", help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default="fedmodels/yolov5/data/hyps/hyp.scratch-low.yaml", help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=1, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    # parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')


    args = parser.parse_args()

    main(args)
