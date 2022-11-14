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

from fedmodels.yolov5.utils.loss import ComputeLoss
from fedmodels.yolov5.utils.torch_utils import smart_optimizer, ModelEMA
from fedmodels.yolov5.utils.general import check_amp, check_img_size
import fedmodels.yolov5.val as validate

from fedutils.data_loader import load_partition_data_custom, load_server_data
from fedutils.util import partial_client_selection, average_model
from fedutils.start_config import init_configure
from fedutils.scheduler import setup_scheduler

import wandb
wandb.login()

def project_conflicting(grads):
    pc_grad = deepcopy(grads)
    # import pdb; pdb.set_trace()
    for g_i in pc_grad:
        random.shuffle(pc_grad)
        for g_j in pc_grad:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

    return merged_grad

def unflatten_grad(grads, shapes):
    unflatten_grad, idx = [], 0
    for shape in shapes:
        length = np.prod(shape)
        unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
        idx += length

    return unflatten_grad

def main(args):
    
    # server participation
    if args.central:
        print(" ========== train with server (central dataset) ========== ")
        name = (
            args.net_name
            + "_lr_"
            + str(args.learning_rate)
            + "_round_"
            + str(args.epoch)
            + "_server_epochs_"
            + str(args.server_local_epoch)
            + "_local_epochs_"
            + str(args.local_epoch)
            + "_optimizer_"
            + str(args.optimizer_type)
            + "_seed_"
            + str(args.seed)
        )
    else:
        print(" ========== train only nodes ========== ")
        name = (
            args.net_name
            + "_lr_"
            + str(args.learning_rate)
            + "_round_"
            + str(args.epoch)
            + "_local_epochs_"
            + str(args.local_epoch)
            + "_optimizer_"
            + str(args.optimizer_type)
            + "_seed_"
            + str(args.seed)
        )
    
    # gradient surgery
    if args.grad_surgery:
        print(" ========== gradient surgery ========== ")
    
    num = 0
    exp_dir = os.path.join(args.save_dir, f"{name}_{num}")
    while os.path.exists(exp_dir):
        num += 1
        exp_dir = os.path.join(args.save_dir, f"{name}_{num}")
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if args.wandb:
        wandb.init(
            project=args.project,
            name=f"{name}_{num}",
            config={
                "learning_rate": args.learning_rate,
                "epoch": args.epoch,
                "server_epoch": args.server_local_epoch,
                "local_epoch": args.local_epoch,
                "batch_size": args.batch_size
            })
        
    # initialization    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, hyp = init_configure(args, device, exp_dir) # default: hyp.scratch-low

    # training, validating, and testing
    ''' prepare dataset '''
    with open(args.data_conf) as f: # default: coco_fl
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    shuffle = args.shuffle
    batch_size, img_size = args.batch_size, args.img_size
    clients, num_clients = args.clients, len(args.clients)
    dataset = load_partition_data_custom(args, hyp, model, data_dict, batch_size, img_size, clients, shuffle)
    [
        train_dataset_dict,
        train_data_num_dict,
        train_data_loader_dict,
        test_loader
    ] = dataset

    server_loader, server_dataset, _ = load_server_data(args, hyp, model, data_dict, batch_size, img_size, shuffle)

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
        imgsz=args.img_size,
        half=True,
        model=model,
        dataloader=test_loader,
        plots=False,
        verbose=False,
        compute_loss=compute_loss,
    )
    
    # log
    # for i in range(len(results_name)):
    wandb.log({f"SERVER_RESULTS_{results_name[3]}": server_results[3]}, step=0)
    for i in range(len(args.clients)):
        wandb.log({f"SERVER_CLIENTS_{i}_CLASS_{args.clients[i]}": server_maps[args.clients[i]]}, step=0)
            
    # Configuration for FedAVG, prepare model, optimizer, and scheduler
    model_all, optimizer_all, scheduler_all, ema_all = partial_client_selection(args, model, hyp)
    model_server = deepcopy(model).cpu() # server for update
    
    # Image size
    gs = max(int(model_server.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(args.img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / args.batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= args.batch_size * accumulate / nbs 
    optimizer_server = smart_optimizer(model_server, args.optimizer_type, args.learning_rate, hyp['momentum'], hyp['weight_decay'])

    lf_server = lambda x: (1 - x / args.server_local_epoch) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler_server = lr_scheduler.LambdaLR(optimizer_server, lr_lambda=lf_server)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    amp_server = check_amp(model_server)    
    scaler_server = torch.cuda.amp.GradScaler(enabled=amp_server)
    
    model_avg = deepcopy(model).cpu() # average server
    optimizer_avg = smart_optimizer(model_avg, args.optimizer_type, args.learning_rate, hyp['momentum'], hyp['weight_decay'])
    
    # model shape
    model_shape = list()
    for group in optimizer_avg.param_groups:
        for p in group['params']:
            model_shape.append(p.shape)
    print("=============== start training ===============")
    for epoch in range(args.epoch):

        # clients_keys
        # clients_data_num

        # Get the quantity of clients joined in the FL train for updating the clients weights
        curr_total_client_lens = 0
        for client in clients_keys:
            curr_total_client_lens += clients_data_num[client]
            
        # if server participate the training
        if args.central:
            curr_total_client_lens += len(server_dataset)

            print("---------- training server ----------")
            # the ratio of clients for updating the clients weights
            server_weight = len(server_dataset) / curr_total_client_lens

            print(
                "train the server", "of communication round", epoch
            )
            nb = len(server_loader)  # number of batches
            nw = max(round(5 * nb), 100) # hyp['warmup_epochs'] = 5
            last_opt_step = -1
            
            model_server = model_server.to(device)
            compute_loss = ComputeLoss(model_server)
            
            server_grad = list()
            for inner_epoch in range(args.server_local_epoch):
                model_server.train()

                pbar = enumerate(server_loader)
                pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
                optimizer_server.zero_grad()
                for step, (imgs, targets, paths, _) in pbar:   
                    ni = step + nb * epoch  # number integrated batches (since train start)
                    imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
                    
                    # Warmup
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        accumulate = max(1, np.interp(ni, xi, [1, nbs / args.batch_size]).round())
                        for j, x in enumerate(optimizer_server.param_groups):
                            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf_server(epoch)])
                            if 'momentum' in x:
                                x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])    
                    
                    # Multi-scale
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                                        
                    with torch.cuda.amp.autocast(amp_server):
                        pred = model_server(imgs)  # forward
                        loss, _ = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    # grad = torch.autograd.grad(loss, model_server.parameters())
                    # import pdb; pdb.set_trace()
                    # if inner_epoch == 0 and step == 0:
                    #     server_grad = deepcopy(grad)
                    # server_grad = tuple(sum(i) for i in zip(grad, server_grad))
                    
                    # Backward
                    scaler_server.scale(loss).backward()
                    # loss.backward()
                    if args.grad_surgery:
                        grad = []
                        for group in optimizer_server.param_groups:
                            for p in group['params']:
                                if p.grad is None:
                                    grad.append(torch.zeros_like(p).cpu())
                                    continue
                                grad.append(p.grad.clone().cpu())
                        server_grad.append(grad)        
                    # optimizer_server.step()                    
                    
                    # Optimize
                    if ni - last_opt_step >= accumulate:
                        scaler.unscale_(optimizer_server)  # unscale gradients
                        torch.nn.utils.clip_grad_norm_(model_server.parameters(), max_norm=10.0)  # clip gradients
                        scaler.step(optimizer_server)  # optimizer.step
                        scaler.update()
                        optimizer_server.zero_grad()
                        last_opt_step = ni
                    
                    wandb.log({"SERVER_LOSS": loss.item()})

                scheduler_server.step()

                server_inner_results, server_inner_maps, _ = validate.run(
                    data_dict,
                    batch_size=args.batch_size,
                    imgsz=args.img_size,
                    half=True,
                    model=model_server,
                    single_cls=False,
                    dataloader=test_loader,
                    plots=False,
                    compute_loss=compute_loss,
                )

                # log
                wandb.log({f"SERVER_INNER_RESULTS_{results_name[3]}": server_inner_results[3]})
                for i in range(len(args.clients)):
                    wandb.log({f"SERVER_INNER_CLIENTS_{i}_CLASS_{args.clients[i]}": server_inner_maps[args.clients[i]]})
         
            print(" ========== finish to train server ========== ")           
        
        clients_weights = {}                            
        lf = lambda x: (1 - x / args.local_epoch) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        # local update
        
        results = np.zeros(
            (num_clients, 7)
        )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        
        node_grad = list()
        for proxy_single_client in range(len(args.clients)):
            print(
                "train the client", proxy_single_client, "of communication round", epoch
            )

            # the ratio of clients for updating the clients weights
            clients_weights[proxy_single_client] = (
                clients_data_num[proxy_single_client] / curr_total_client_lens
            )

            # client's train dataset 
            train_loader = train_data_loader_dict[proxy_single_client]
            
            model = model_all[proxy_single_client].to(device)
            import pdb; pdb.set_trace()
            optimizer = optimizer_all[proxy_single_client]
            scheduler = scheduler_all[proxy_single_client]
            ema = ema_all[proxy_single_client]
            compute_loss = ComputeLoss(model)
                        
            nb = len(train_loader)  # number of batches
            nw = max(round(5 * nb), 100) # hyp['warmup_epochs'] = 5
            scheduler.last_epoch = -1
            last_opt_step = -1

            amp = check_amp(model)    
            scaler = torch.cuda.amp.GradScaler(enabled=amp)
            
            single_node_grad = list()
            for inner_epoch in range(args.local_epoch):
                model.train()
                
                pbar = enumerate(train_loader)
                pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
                
                optimizer.zero_grad()
                
                for step, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                    ni = step + nb * epoch  # number integrated batches (since train start)
                    imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                    # Warmup
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        accumulate = max(1, np.interp(ni, xi, [1, nbs / args.batch_size]).round())
                        for j, x in enumerate(optimizer.param_groups):
                            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                            if 'momentum' in x:
                                x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])    
                    
                    # Multi-scale
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                                        
                    with torch.cuda.amp.autocast(amp):
                        pred = model(imgs)  # forward
                        loss, _ = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

                    # grad = torch.autograd.grad(loss, model.parameters())
                    # if inner_epoch == 0 and step == 0:
                    #     grad_sum = deepcopy(grad)
                    # grad_sum = [sum(i) for i in zip(grad, grad_sum)]
    
                    # Backward
                    scaler.scale(loss).backward()
                    # loss.backward()
                    
                    if args.grad_surgery:
                        grad = []
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is None:
                                    grad.append(torch.zeros_like(p))
                                    continue
                                grad.append(p.grad.clone().cpu())
                        single_node_grad.append(grad)        
                    # optimizer.step()
                    
                    # Optimize
                    if ni - last_opt_step >= accumulate:
                        scaler.unscale_(optimizer)  # unscale gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                        scaler.step(optimizer)  # optimizer.step
                        scaler.update()
                        optimizer.zero_grad()
                        
                        ema.update(model)
                        last_opt_step = ni

                    wandb.log({f"NODE_{proxy_single_client}_LOSS": loss.item()})
                        
                scheduler.step()
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            
                results[proxy_single_client], maps, _ = validate.run(
                    data_dict,
                    batch_size=args.batch_size,
                    imgsz=args.img_size,
                    half=True,
                    model=ema.ema,
                    single_cls=False,
                    dataloader=test_loader,
                    plots=False,
                    compute_loss=compute_loss,
                )
                
                
                # log
                wandb.log({f"CLIENTS_{proxy_single_client}_INNER_RESULTS_{results_name[3]}": results[proxy_single_client][3]})
                for i in range(len(args.clients)):
                    wandb.log({f"CLIENTS_{proxy_single_client}_INNER_CLASS_{args.clients[i]}": maps[args.clients[i]]})
            # import pdb; pdb.set_trace()
            # node_grad.append([i.cpu() for i in grad_sum])
            
            # log
            wandb.log({f"CLIENTS_{proxy_single_client}_RESULTS_{results_name[3]}": results[proxy_single_client][3]})

            for i in range(len(args.clients)):
                wandb.log({f"CLIENTS_{proxy_single_client}_CLASS_{args.clients[i]}": maps[args.clients[i]]})

            # grad
            node_grad.append(single_node_grad)
        # import pdb; pdb.set_trace()
            
        """ ---- model average and eval ---- """
        if args. grad_surgery:
            if args.central:
                optimizer_avg.zero_grad()
                node_grad.append(server_grad)
                node_grad = [torch.from_numpy(np.sum(grad, axis=0)) for grad in node_grad]
                flatten_grad = torch.cat([g.flatten() for g in node_grad])
                merged_grads = project_conflicting(flatten_grad)
                unflatten_grads = unflatten_grad(merged_grads, model_shape)

                idx = 0
                for group in optimizer_avg.param_groups:
                    for p in group['params']:
                        # if p.grad is None: continue
                        p.grad = unflatten_grads[idx]
                        idx += 1
                optimizer_avg.step()
            else:
                optimizer_avg.zero_grad()
                node_grad = [torch.from_numpy(np.sum(grad, axis=0)) for grad in node_grad]
                # import pdb; pdb.set_trace()
                flatten_grad = torch.cat([g.flatten() for g in node_grad])
                merged_grads = project_conflicting(node_grad)
                unflatten_grads = unflatten_grad(merged_grads, model_shape)

                idx = 0
                for group in optimizer_avg.param_groups:
                    for p in group['params']:
                        # if p.grad is None: continue
                        p.grad = unflatten_grads[idx]
                        idx += 1
                optimizer_avg.step()
        else:
            if args.central:
                average_model(args, model_avg, model_all, model_server, server_weight, clients_weights)
            else:
                weight=None
                average_model(args, model_avg, model_all, model_server, weight, clients_weights)
        
        # then evaluate server
        model_avg.to(device)
        compute_loss = ComputeLoss(model_avg)
        server_results, server_maps, _ = validate.run(
            data_dict,
            batch_size=args.batch_size,
            imgsz=args.img_size,
            half=True,
            model=model_avg,
            single_cls=False,
            dataloader=test_loader,
            plots=False,
            compute_loss=compute_loss,
        )
        
        # for i in range(len(results_name)):
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
    parser.add_argument("--num_workers", default=8, type=int, help="num_workers")
    parser.add_argument(
        "--weight_decay",
        default=0,
        choices=[0.05, 0],
        type=float,
        help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper",
    )
    parser.add_argument(
        "--grad_clip",
        action="store_true",
        default=True,
        help="whether gradient clip to 1 or not",
    )
    parser.add_argument(
        "--img_size", default=640, type=int, help="Final train resolution"
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Local batch size for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )  # 99999

    ## section 2:  DL learning rate related
    parser.add_argument(
        "--decay_type",
        choices=["cosine", "linear", "step"],
        default="step",
        help="How to decay the learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=100,
        type=int,
        help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.",
    )
    parser.add_argument(
        "--step_size",
        default=30,
        type=int,
        help="Period of learning rate decay for step size learning rate decay",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-6,
        type=float,
        help="The initial learning rate for SGD.",
    )

    ## FL related parameters
    parser.add_argument(
        "--local_epoch", default=1, type=int, help="Local training epoch in FL"
    )
    parser.add_argument(
        "--server_local_epoch", default=1, type=int, help="Local training epoch in FL"
    )
    parser.add_argument(
        "--epoch",
        default=50,
        type=int,
        help="Total communication rounds",
    )
    parser.add_argument(
        "--clients", nargs="+", default=[56], help="56: chair, 60: table"
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

    ## YOLO hyperparameters
    parser.add_argument(
        "--weights",
        type=str,
        default="/hdd/hdd3/coco_fl/best.pt",
        help="initial weights path",
    )
    parser.add_argument(
        "--yolo_cfg",
        type=str,
        default="fedmodels/yolov5/models/yolov5s.yaml",
        help="model.yaml path",
    )
    parser.add_argument(
        "--data_conf",
        type=str,
        default="data/coco_custom.yaml",
        help="dataset.yaml path",
    )
    parser.add_argument(
        "--yolo_hyp",
        type=str,
        default="fedmodels/yolov5/data/hyps/hyp.scratch-low.yaml",
        help="hyperparameters path",
    )
    parser.add_argument("--shuffle", default=False, help="dataset shuffle.")

    # saving ------------------
    parser.add_argument("--save_dir", default="./exp/")
    parser.add_argument("--wandb", default=True)
    parser.add_argument("--project", default="FedYOLO")

    args = parser.parse_args()

    main(args)
