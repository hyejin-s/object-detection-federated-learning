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
    message = ''

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += "================ FL train of %s with total model parameters: %2.1fM  ================\n" % (args.FL_platform, num_params)

    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'

    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '


    ## save to disk of current log

    args.file_name = os.path.join(args.output_dir, 'log_file.txt')

    with open(args.file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)

def initization_configure(args, vis= False):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not args.device == 'cpu':
        torch.cuda.manual_seed(args.seed)

    # Set model type related parameters
    if "ResNet" in args.FL_platform:
        args.Use_ResNet = True
        if '101' in args.net_name:
            model = torch_models.resnet152(pretrained=args.Pretrained)
            # model.fc = nn.Linear(2048, args.num_classes)
            print('We use ResNet 152')

        elif '32_8' in args.net_name:

            model = torch_models.resnext101_32x8d(pretrained=args.Pretrained)
            print('We use ResNet 101-32*8d')

        else:
            model = torch_models.resnet50(pretrained=args.Pretrained)
            print('We use default ResNet 50')
        model.fc = nn.Linear(model.fc.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "ViT" in args.FL_platform:
        if 'tiny' in args.net_name:
            print('We use ViT tiny')
            from timm.models.vision_transformer import vit_tiny_patch16_224
            model = vit_tiny_patch16_224(pretrained=args.Pretrained)
        elif 'small' in args.net_name:
            print('We use ViT small')
            from timm.models.vision_transformer import vit_small_patch16_224
            model = vit_small_patch16_224(pretrained=args.Pretrained)
        else:
            from timm.models.vision_transformer import vit_base_patch16_224
            print('We use default ViT settting base')
            model = vit_base_patch16_224(pretrained=args.Pretrained)

        model.head = Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)

        # we test with timm

    elif "Swin" in args.FL_platform:
        print('We use Swin')
        if not args.cfg:
            sys.exit('Network configure file cfg for Swin is missing, code is exit')
        swin_args = get_config(args)
        model = build_model(args, swin_args)
        if args.Pretrained:
            checkpoint = torch.load(args.pretrained_dir, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)

        model.head = Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "YOLOv5" in args.FL_platform:
        print('We use YOLOv5')
        # Model
        model, args = init_yolo(args=args, device=args.device)
        # check_suffix(weights, '.pt')  # check weights
        # pretrained = weights.endswith('.pt')
        # if pretrained:
        #     with torch_distributed_zero_first(LOCAL_RANK):
        #         weights = attempt_download(weights)  # download if not found locally
        #     ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        #     model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        #     exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        #     csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        #     csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        #     model.load_state_dict(csd, strict=False)  # load
        #     LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        # else:
        #     model = Model(cfg, ch=3, nc=args.num_classes, anchors=hyp.get('anchors')).to(device)  # create

    # set output parameters
    print(args.optimizer_type)
    args.name = args.net_name + '_lr_' + str(args.learning_rate) + '_Pretrained_' \
                + str(args.Pretrained) + "_optimizer_" + str(args.optimizer_type) +  '_WUP_'  + str(args.warmup_steps) \
                + '_Round_' + str(args.max_communication_rounds) + '_local_epochs_' + str(args.local_epoch) + '_Seed_' + str(args.seed)

    args.output_dir = os.path.join('output', args.FL_platform, args.dataset, args.name)
    os.makedirs(args.output_dir, exist_ok=True)

    print_options(args, model)

    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}

    return model


