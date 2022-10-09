import os
from tqdm import tqdm
import random
import argparse


class CocoCustomizer:
    def __init__(self, data_path, dataset="train2017"):
        self.data_path = data_path
        self.dataset = dataset
        self.dir = os.path.join(
            data_path, dataset
        )  # e.g. '/hdd/hdd3/coco/labels/train2017'
        self.file_list = os.listdir(self.dir)

    def read_file(self, file_name):
        """
        Given .txt file, read the file line by line
        return : list of str ('class# x y w h')
        """
        f = open(os.path.join(self.dir, file_name), "r")
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        f.close()

        return lines

    def class_removal(self, file_name, remove_class: list, keep_prob=0.1):
        """
        Remove classes in input list
        Some class instances are not removed with a probability if 'keep_prob'.
        return: list of str ('class coordinate')
        """
        obj_list = self.read_file(file_name)
        removed_list = obj_list.copy()

        for object in obj_list:
            # object: 'class# x y w h'
            # e.g. '58 0.911730 0.533705 0.176540 0.559554'
            if int(object.split()[0]) in remove_class:
                if random.random() > keep_prob:
                    removed_list.remove(object)

        return removed_list

    def customize_data(self, remove_class: list, keep_prob, save_root):
        """
        Remove the class in total dataset.
        Keep it with a probability of keep_prob.
        """
        save_dir = os.path.join(save_root, self.dataset)
        create_dir(save_dir)
        for file in tqdm(self.file_list):
            after_remove_list = self.class_removal(file, remove_class, keep_prob)
            with open(os.path.join(save_dir, file), "w+") as f:
                for obj in after_remove_list:
                    f.write(obj)
                    f.write("\n")
        print(f"{self.dir} done")


def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory." + directory)


def main(args):
    for dataset in ["train2017", "val2017"]:
        customizer = CocoCustomizer(args.data_path, dataset)
        customizer.customize_data(args.remove_class, args.keep_prob, args.save_root)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/hdd/hdd3/coco/labels")
    parser.add_argument(
        "--save_root",
        help="save root",
        type=str,
        default="/hdd/hdd3/coco_no_chair_table/labels",
    )
    parser.add_argument("--keep_prob", help="keep ratio", type=float, default=0.1)
    parser.add_argument(
        "--remove_class",
        help="remove list, 56: chair, 60: dining table",
        default=[56, 60],
    )
    args = parser.parse_args()

    main(args)
