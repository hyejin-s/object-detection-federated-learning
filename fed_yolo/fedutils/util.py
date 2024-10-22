from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml

import torch

from fedutils.scheduler import setup_scheduler
from fedmodels.yolov5.utils.loss import ComputeLoss
from fedmodels.yolov5.utils.torch_utils import smart_optimizer, ModelEMA
from torch.optim import lr_scheduler

from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == "sgd":
        optimizer = optim.SGD(
            parameters,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            nesterov=True,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(
            parameters,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=config.TRAIN.OPTIMIZER.BETAS,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, "module") else model
    client_name = os.path.basename(args.single_client).split(".")[0]
    model_checkpoint = os.path.join(
        args.output_dir, "%s_%s_checkpoint.bin" % (args.name, client_name)
    )

    torch.save(model_to_save.state_dict(), model_checkpoint)
    # print("Saved model checkpoint to [DIR: %s]", args.output_dir)


def inner_valid(args, model, test_loader):
    eval_losses = AverageMeter()
    compute_loss = ComputeLoss(model)

    print("++++++ Running Validation of client", args.single_client, "++++++")
    model.eval()
    all_preds, all_label = [], []

    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        # batch = tuple(t.to(args.device) for t in batch)
        x, y = batch[0].float().to(args.device), batch[1].float().to(args.device)
        with torch.no_grad():
            logits = model(x)

            if args.num_classes > 1:
                eval_loss, loss_items = compute_loss(logits, y)
                eval_losses.update(eval_loss.item())

            if args.num_classes > 1:
                preds = torch.argmax(logits, dim=-1)
            else:
                preds = logits

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
    all_preds, all_label = all_preds[0], all_label[0]
    if not args.num_classes == 1:
        eval_result = simple_accuracy(all_preds, all_label)
    else:
        # eval_result =  mean_absolute_error(all_preds, all_label)
        eval_result = mean_squared_error(all_preds, all_label)

    model.train()

    return eval_result, eval_losses


def metric_evaluation(args, eval_result):
    if args.num_classes == 1:
        if args.best_acc[args.single_client] < eval_result:
            Flag = False
        else:
            Flag = True
    else:
        if args.best_acc[args.single_client] < eval_result:
            Flag = True
        else:
            Flag = False
    return Flag


def valid(args, model, test_loader=None, TestFlag=False):
    # Validation!
    eval_result, eval_losses = inner_valid(args, model, test_loader)

    print("Valid Loss: %2.5f" % eval_losses.avg, "Valid metric: %2.5f" % eval_result)
    if args.dataset == "CelebA":
        if args.best_eval_loss[args.single_client] > eval_losses.val:
            # if args.best_acc[args.single_client] < eval_result:
            if args.save_model_flag:
                save_model(args, model)

            args.best_acc[args.single_client] = eval_result
            args.best_eval_loss[args.single_client] = eval_losses.val
            print(
                "The updated best metric of client",
                args.single_client,
                args.best_acc[args.single_client],
            )

            if TestFlag:
                test_result, eval_losses = inner_valid(args, model, test_loader)
                args.current_test_acc[args.single_client] = test_result
                print(
                    "We also update the test acc of client",
                    args.single_client,
                    "as",
                    args.current_test_acc[args.single_client],
                )
        else:
            print(
                "Donot replace previous best metric of client",
                args.best_acc[args.single_client],
            )
    else:  # we use different metrics
        # if args.best_acc[args.single_client] < eval_result:
        if metric_evaluation(args, eval_result):
            if args.save_model_flag:
                save_model(args, model)

            args.best_acc[args.single_client] = eval_result
            args.best_eval_loss[args.single_client] = eval_losses.val
            print(
                "The updated best metric of client",
                args.single_client,
                args.best_acc[args.single_client],
            )

            if TestFlag:
                test_result, eval_losses = inner_valid(args, model, test_loader)
                args.current_test_acc[args.single_client] = test_result
                print(
                    "We also update the test acc of client",
                    args.single_client,
                    "as",
                    args.current_test_acc[args.single_client],
                )
        else:
            print(
                "Donot replace previous best metric of client",
                args.best_acc[args.single_client],
            )

    args.current_acc[args.single_client] = eval_result


def optimization_fun(args, model):

    # Prepare optimizer, scheduler
    if args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            eps=1e-8,
            betas=(0.9, 0.999),
            lr=args.learning_rate,
            weight_decay=0.05,
        )

    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            eps=1e-8,
            betas=(0.9, 0.999),
            lr=args.learning_rate,
            weight_decay=0.05,
        )

        print(
            "===============Not implemented optimization type, we used default adamw optimizer ==============="
        )
    return optimizer


def partial_client_selection(args, model, hyp):

    # # Select partial clients join in FL train
    # if args.num_local_clients == -1:  # all the clients joined in the train
    #     proxy_clients = args.dis_cvs_files
    #     args.num_local_clients = len(
    #         args.dis_cvs_files
    #     )  # update the true number of clients
    # else:
    #     args.proxy_clients = ["train_" + str(i) for i in range(args.num_local_clients)]

    # Generate model for each client
    model_all = {}
    optimizer_all = {}
    scheduler_all = {}
    ema_all = {}

    for proxy_single_client in range(len(args.clients)):
        model_all[proxy_single_client] = deepcopy(model).cpu()
        ema_all[proxy_single_client] = ModelEMA(deepcopy(model))

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(
            round(nbs / args.batch_size), 1
        )  # accumulate loss before optimizing
        hyp["weight_decay"] *= args.batch_size * accumulate / nbs
        optimizer_all[proxy_single_client] = smart_optimizer(
            model_all[proxy_single_client],
            "SGD",
            args.learning_rate,
            hyp["momentum"],
            hyp["weight_decay"],
        )

        lf = (
            lambda x: (1 - x / args.local_epoch) * (1.0 - hyp["lrf"]) + hyp["lrf"]
        )  # linear
        scheduler_all[proxy_single_client] = lr_scheduler.LambdaLR(
            optimizer_all[proxy_single_client], lr_lambda=lf
        )

    return model_all, optimizer_all, scheduler_all, ema_all


def average_model(
    args, model_avg, model_all, model_server, server_weight, clients_weights, device
):
    print("---- calculate the model avg ----")
    global_params = model_avg.state_dict()

    for client in range(len(args.clients)):
        single_client_weight = clients_weights[client]
        single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()
        net_params = model_all[client].state_dict()
        if client == 0:
            for key in net_params:
                global_params[key] = net_params[key] * single_client_weight
        else:
            for key in net_params:
                global_params[key] += net_params[key] * single_client_weight

    model_avg.load_state_dict(global_params)
    model_avg.to(device)

    # if args.central:
    #     server_weight = torch.from_numpy(np.array(server_weight)).float()
    #     tmp_param_data = (
    #         tmp_param_data
    #         + dict(model_server.named_parameters())[name].data * server_weight
    #     )

    print("---- update each client model parameters ----")
    for client in range(len(args.clients)):
        model_all[client].load_state_dict(global_params)

    # if args.central:
    #     tmp_params = dict(model_server.named_parameters())
    #     for name, param in params.items():
    #         tmp_params[name].data.copy_(param.data)
