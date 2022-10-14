import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import argparse
import numpy as np
import shutil


def main(args):
    partition = "labels/train2017/"
    img_partition = "images/train2017/"
    coco_path = os.listdir(os.path.join(args.data_dir, partition))

    coco_num = len(coco_path)

    copy_num = int(args.ratio * coco_num)
    random_list = random.sample(coco_path, copy_num)
    # print(random_list)

    img_path = os.path.join(args.save_dir, f"images/random_{args.ratio}/")
    label_path = os.path.join(args.save_dir, f"labels/random_{args.ratio}/")

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if not os.path.exists(label_path):
        os.makedirs(label_path)

    for file in random_list:
        file_name = file[:-4]

        # image copy
        img_path = os.path.join(args.save_dir, f"images/random_{args.ratio}/")
        shutil.copy(
            args.data_dir + img_partition + file_name + ".jpg",
            os.path.join(img_path, file_name) + ".jpg",
        )
        # label copy
        label_path = os.path.join(args.save_dir, f"labels/random_{args.ratio}/")
        shutil.copy(
            args.data_dir + partition + file_name + ".txt",
            os.path.join(label_path, file_name) + ".txt",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        help="save root",
        type=str,
        default="/hdd/hdd3/coco_custom/",
    )
    parser.add_argument("--data_dir", type=str, default="/hdd/hdd3/coco/")
    parser.add_argument(
        "--ratio",
        help="copy ratio",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()
    main(args)
