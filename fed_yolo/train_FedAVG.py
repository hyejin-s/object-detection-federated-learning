# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
from pickle import TRUE
import warnings
import argparse
import numpy as np
from copy import deepcopy
import yaml

import torch
from torch.utils.tensorboard import SummaryWriter

from fedmodels.yolov5.utils.loss import ComputeLoss
import fedmodels.yolov5.val as validate

from fedutils.data_loader import load_partition_data_custom, load_server_data
from fedutils.util import Partial_Client_Selection, average_model, optimization_fun
from fedutils.start_config import init_configure
from fedutils.scheduler import setup_scheduler


def train(args, model):
    """train the model"""
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # prepare dataset ----------
    with open(args.yolo_hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if "box" not in hyp:
            warnings.warn(
                'Compatibility: %s missing "box" which was renamed from "giou" in %s'
                % (args.yolo_hyp, "https://github.com/ultralytics/yolov5/pull/1120")
            )
            hyp["box"] = hyp.pop("giou")

    dataset = load_partition_data_custom(args, hyp, model, args.class_list)
    [
        train_dataset_dict,
        train_data_num_dict,
        train_data_loader_dict,
        test_data_loader_dict,
        val_loader,
        args.num_classes,
    ] = dataset

    server_loader, server_dataset, _, _ = load_server_data(args, hyp, model)

    args.dis_cvs_files = list(train_dataset_dict.keys())
    args.clients_with_len = {
        name: train_data_num_dict[name] for name in args.dis_cvs_files
    }

    # model.to(args.device)
    compute_loss = ComputeLoss(model)

    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    # save file path ----------
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    num = 1
    exp_dir = os.path.join(args.save_dir, f"exp{num}")
    while os.path.exists(exp_dir):
        num += 1
        exp_dir = os.path.join(args.save_dir, f"exp{num}")

    # import pdb; pdb.set_trace()
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    with open(f"{exp_dir}/server.txt", "a+") as f:
        f.write(f"P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)")
        f.write("\n")

    with open(f"{exp_dir}/clients.txt", "a+") as f:
        f.write(f"P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)")
        f.write("\n")

    # checking initial model performace
    initial_results, _, _ = validate.run(
        data_dict,
        weights=args.weights,
        batch_size=args.batch_size,
        imgsz=args.img_size,
        half=True,
        # model=model,
        dataloader=val_loader,
        plots=False,
        verbose=False,
        compute_loss=compute_loss,
    )

    with open(f"{exp_dir}/server.txt", "a+") as f:
        f.write(str(initial_results))
        f.write("\n")

    # Configuration for FedAVG, prepare model, optimizer, scheduler
    model_all, optimizer_all, scheduler_all = Partial_Client_Selection(args, model)
    model_server = deepcopy(model)  # .cpu()  # server
    optimizer_server = optimization_fun(args, model_server)
    scheduler_server = setup_scheduler(args, optimizer_server, 1)

    model_avg = deepcopy(model)  # .cpu()  # server

    # train
    print("=============== running training ===============")
    compute_loss = ComputeLoss(model)
    total_clients = args.dis_cvs_files
    epoch = -1

    while True:
        epoch += 1
        # randomly select partial clients
        if args.num_local_clients == len(args.dis_cvs_files):
            # just use all the local clients
            curr_selected_clients = args.proxy_clients
        else:
            curr_selected_clients = np.random.choice(
                total_clients, args.num_local_clients, replace=False
            ).tolist()

        # Get the quantity of clients joined in the FL train for updating the clients weights
        curr_total_client_lens = 0
        for client in curr_selected_clients:
            curr_total_client_lens += args.clients_with_len[client]
        if args.parti_server:
            curr_total_client_lens += len(server_dataset)

        # local update
        for curr_single_client, proxy_single_client in zip(
            curr_selected_clients, args.proxy_clients
        ):
            args.single_client = curr_single_client

            # the ratio of clients for updating the clients weights
            args.clients_weightes[proxy_single_client] = (
                args.clients_with_len[curr_single_client] / curr_total_client_lens
            )

            train_loader = train_data_loader_dict[proxy_single_client]

            model = model_all[proxy_single_client].train()  # .to(args.device).train()
            compute_loss = ComputeLoss(model)
            optimizer = optimizer_all[proxy_single_client]
            scheduler = scheduler_all[proxy_single_client]
            if args.decay_type == "step":
                scheduler.step()

            print(
                "Train the client", curr_single_client, "of communication round", epoch
            )

            for inner_epoch in range(args.local_epoch):
                for step, batch in enumerate(train_loader):
                    args.global_step_per_client[proxy_single_client] += 1

                    x, y = batch[0].float().to(args.device), batch[1].float().to(
                        args.device
                    )

                    optimizer.zero_grad()
                    pred = model(x)

                    loss, _ = compute_loss(pred, y)  # loss scaled by batch_size
                    loss.backward()

                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    optimizer.step()
                    if not args.decay_type == "step":
                        scheduler.step()

                    # tensorboard
                    writer.add_scalar(
                        str(proxy_single_client) + "/lr",
                        scalar_value=optimizer.param_groups[0]["lr"],
                        global_step=args.global_step_per_client[proxy_single_client],
                    )
                    writer.add_scalar(
                        str(proxy_single_client) + "/loss",
                        scalar_value=loss.item(),
                        global_step=args.global_step_per_client[proxy_single_client],
                    )

                    args.learning_rate_record[proxy_single_client].append(
                        optimizer.param_groups[0]["lr"]
                    )

                    if (step) % 10 == 0:
                        print(
                            "client",
                            curr_single_client,
                            step,
                            "/",
                            len(train_loader),
                            "inner epoch",
                            inner_epoch,
                            "round",
                            epoch,
                            "/",
                            args.max_communication_rounds,
                            "loss",
                            loss.item(),
                            "lr",
                            optimizer.param_groups[0]["lr"],
                        )
        weight = None

        if args.parti_server:
            print("----- training server -----")
            args.single_client = curr_single_client

            # the ratio of clients for updating the clients weights
            server_weight = len(server_dataset) / curr_total_client_lens

            model = model_server.train()  # .to(args.device).train()
            compute_loss = ComputeLoss(model)

            if args.decay_type == "step":
                scheduler_server.step()

            print("Train the server", "of communication round", epoch)

            for inner_epoch in range(args.server_local_epoch):
                for step, batch in enumerate(server_loader):
                    x, y = batch[0].float().to(args.device), batch[1].float().to(
                        args.device
                    )

                    optimizer_server.zero_grad()
                    pred = model(x)

                    loss, _ = compute_loss(pred, y)  # loss scaled by batch_size
                    loss.backward()

                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    optimizer_server.step()
                    if not args.decay_type == "step":
                        scheduler_server.step()

                    args.learning_rate_record[proxy_single_client].append(
                        optimizer_server.param_groups[0]["lr"]
                    )

                    if (step) % 10 == 0:
                        print(
                            "server",
                            step,
                            "/",
                            len(server_loader),
                            "inner epoch",
                            inner_epoch,
                            "round",
                            epoch,
                            "/",
                            args.max_communication_rounds,
                            "loss",
                            loss.item(),
                            "lr",
                            optimizer_server.param_groups[0]["lr"],
                        )

            # we use frequent transfer of model between GPU and CPU due to limitation of GPU memory
            # model.to('cpu')

        """ ---- model average and eval ---- """

        # then evaluate per clients
        results = np.zeros(
            (args.client_num_in_total, 7)
        )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

        for curr_single_client, proxy_single_client in zip(
            curr_selected_clients, args.proxy_clients
        ):
            args.single_client = curr_single_client
            model = model_all[proxy_single_client]
            model.to("cuda:0")
            compute_loss = ComputeLoss(model)
            results[proxy_single_client], maps, _ = validate.run(
                data_dict,
                batch_size=args.batch_size,
                imgsz=args.img_size,
                half=True,
                model=model,
                single_cls=False,
                dataloader=val_loader,
                plots=False,
                compute_loss=compute_loss,
            )

        # evaluate fine-tuning server's result
        if args.parti_server:
            # model_server.to(args.device)
            compute_loss = ComputeLoss(model_server)
            results_server, maps, _ = validate.run(
                data_dict,
                batch_size=args.batch_size,
                imgsz=args.img_size,
                half=True,
                model=model_server,
                single_cls=False,
                dataloader=val_loader,
                plots=False,
                compute_loss=compute_loss,
            )

            with open(f"{exp_dir}/server_fine_tuning.txt", "a+") as f:
                f.write(str(results_server))
                f.write("\n")

        # average model
        if args.parti_server:
            average_model(args, model_avg, model_all, model_server, server_weight)
        else:
            average_model(args, model_avg, model_all, model_server, weight)

        # then evaluate server
        # model_avg.to(args.device)
        compute_loss = ComputeLoss(model_avg)
        results_avg_server, maps, _ = validate.run(
            data_dict,
            batch_size=args.batch_size,
            imgsz=args.img_size,
            half=True,
            model=model_avg,
            single_cls=False,
            dataloader=val_loader,
            plots=False,
            compute_loss=compute_loss,
        )

        with open(f"{exp_dir}/server.txt", "a+") as f:
            f.write(str(results_avg_server))
            f.write("\n")

        with open(f"{exp_dir}/clients.txt", "a+") as f:
            for curr_single_client, proxy_single_client in zip(
                curr_selected_clients, args.proxy_clients
            ):
                f.write(f"{proxy_single_client}: {str(results[proxy_single_client])}")
                f.write("\n")

        # writer.add_scalar("test/average_accuracy", scalar_value=np.asarray(tmp_round_acc).mean(), global_step=epoch)
        if (
            args.global_step_per_client[proxy_single_client]
            >= args.t_total[proxy_single_client]
        ):
            break

    writer.close()
    print("================ end training ================ ")


def main(args):

    if args.parti_server:
        print("Train with server")
    else:
        print("Train only nodes")
    # Initialization
    model = init_configure(args)
    # print(model)

    # Training, Validating, and Testing
    train(args, model)

    message = "\n \n ============== final performance ================= \n"
    message += "Final union test accuracy is: %2.5f  \n" % (
        np.asarray(list(args.current_test_acc.values())).mean()
    )
    message += "================ end ================ \n"

    with open(args.file_name, "a+") as args_file:
        args_file.write(message)
        args_file.write("\n")

    print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General DL parameters
    parser.add_argument(
        "--net_name",
        type=str,
        default="Yolov5s-FedAVG",
        help="Basic Name of this run with detailed network-architecture selection. ",
    )
    parser.add_argument(
        "--FL_platform",
        type=str,
        default="YOLOv5-FedAVG",
        choices=[
            "YOLOv5-FedAVG",
        ],
        help="Choose of different FL platform.",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "per_class"],
        default="per_class",
        help="Node; Which dataset.",
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/", help="Where is dataset located."
    )

    parser.add_argument(
        "--save_model_flag",
        action="store_true",
        default=False,
        help="Save the best model for each client.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/swin_tiny_patch4_window7_224.yaml",
        metavar="FILE",
        help="path to args file for Swin-FL",
    )

    parser.add_argument(
        "--Pretrained",
        action="store_true",
        default=True,
        help="Whether use pretrained or not",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default="checkpoint/swin_tiny_patch4_window7_224.pth",
        help="Where to search for pretrained ViT models. [ViT-B_16.npz,  imagenet21k+imagenet2012_R50+ViT-B_16.npz]",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="The output directory where checkpoints/results/logs will be written.",
    )
    parser.add_argument(
        "--optimizer_type",
        default="sgd",
        choices=["sgd", "adamw"],
        type=str,
        help="Ways for optimization.",
    )
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
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
    # parser.add_argument("--gpu_ids", type=str, default='1,2,3', help="gpu ids: e.g. 0  0,1,2")

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )  # 99999

    ## section 2:  DL learning rate related
    parser.add_argument(
        "--decay_type",
        choices=["cosine", "linear", "step"],
        default="cosine",
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
        default=1e-3,
        type=float,
        help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT",
    )
    # parser.add_argument("--learning_rate", default=3e-2, type=float, choices=[5e-4, 3e-2, 1e-3],  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    # 1e-5 for ViT central

    ## FL related parameters
    parser.add_argument(
        "--local_epoch", default=1, type=int, help="Local training epoch in FL"
    )
    parser.add_argument(
        "--server_local_epoch",
        default=1,
        type=int,
        help="Server local training epoch in FL",
    )
    parser.add_argument(
        "--max_communication_rounds",
        default=100,
        type=int,
        help="Total communication rounds",
    )
    parser.add_argument(
        "--num_local_clients",
        default=-1,
        choices=[10, -1],
        type=int,
        help="Num of local clients joined in each FL train. -1 indicates all clients",
    )
    # parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "real", "central"], default="split_3", help="Which data partitions to use")
    parser.add_argument("--client_num_in_total", type=int, default=2, help=",,")
    parser.add_argument(
        "--class_list", nargs="+", default=[56, 60], help="56: chair, 60: table"
    )

    parser.add_argument(
        "--parti_server",
        default=True,
        help="whether server participate training with server dataset",
    )

    ## YOLO hyperparameters
    parser.add_argument(
        "--weights",
        type=str,
        default="/hdd/hdd3/coco_custom/no_chair_table_best.pt",
        help="initial weights path",
    )
    parser.add_argument(
        "--yolo_cfg",
        type=str,
        default="./fedmodels/yolov5/models/yolov5s.yaml",
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
    parser.add_argument("--img_size_list", default=[640, 640])
    parser.add_argument("--yolo_bs", default=32, type=int, help="dataset batch size.")
    parser.add_argument("--shuffle", default=False, help="dataset shuffle.")

    # saving ------------------
    parser.add_argument("--save_dir", default="./output/")

    args = parser.parse_args()

    main(args)
