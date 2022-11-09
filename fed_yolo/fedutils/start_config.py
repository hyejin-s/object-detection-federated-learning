import os
import random
import numpy as np
import yaml

import torch
from fedmodels.init_yolo import init_yolo

def init_configure(args, device, dir):
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if not device == "cpu":
        torch.cuda.manual_seed(args.seed)
    model, hyp = init_yolo(args=args, device=device)

    # set output parameters
    print(args.optimizer_type)
    message = ""

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += (
        "================ FL train with total model parameters: %2.1fM  ================\n"
        % (num_params)
    )

    message += "++++++++++++++++ Other Train related parameters ++++++++++++++++ \n"

    for k, v in sorted(vars(args).items()):
        comment = ""
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "++++++++++++++++  End of show parameters ++++++++++++++++ "
    
    with open(os.path.join(dir, "log.txt"), "wt") as f:
        f.write(message)
        f.write("\n")
    
    # save run settings
    with open(os.path.join(dir, "hyp.yaml"), "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
        
    # save args as yaml
    with open(os.path.join(dir, "opt.yaml"), "w") as f:
        yaml.dump(args.__dict__, f, sort_keys=False)
        
    print(message)

    return model, hyp
