import os
from tqdm import tqdm
import random
import argparse
import numpy as np
import shutil


class CocoCustomizeExcept:
    def __init__(self, data_path, partition="train2017"):
        self.data_path = data_path
        self.dir = os.path.join(data_path, partition)
        self.file_list = os.listdir(self.dir)

    def create_dir(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Creating directory." + directory)

    def read_file(self, file_name):
        """
        Given .txt file, read the file line by line
        return: list of str ('class coordinate')
        """
        f = open(os.path.join(self.dir, file_name), "r")
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        f.close()

        return lines

    def class_removal(self, file_name, exp_list: list):
        """
        Remove classes except exp_list
        return: list of str ('class coordinate')
        """
        obj_list = self.read_file(file_name)
        obj_leave_list = obj_list.copy()

        for object in obj_list:
            if int(object.split()[0]) not in exp_list:
                obj_leave_list.remove(object)

        return obj_leave_list

    def collect_data(self, data_dir, save_dir, collect_class_list: list):
        """
        Remove the class except collect_class_list in total dataset.
        Then, save that imgs and labels.
        """
        for file in tqdm(self.file_list):  ## labels
            after_remove_list = self.class_removal(
                file, collect_class_list
            )  ### check class in file

            if len(after_remove_list) != 0:  ### if remain
                obj_list = list()
                for i in range(len(after_remove_list)):
                    obj_class = int(after_remove_list[i].split()[0])
                    obj_list.append(obj_class)
                obj_list_uniq = np.unique(obj_list)

                """ save imgs and labels for each classes """
                for obj in obj_list_uniq:
                    labels_list = [
                        i for i in after_remove_list if int(i.split()[0]) == obj
                    ]
                    label_path = os.path.join(save_dir, f"labels/train_class{obj}/")
                    self.create_dir(label_path)
                    with open(f"{label_path}{file}", "w+") as f:
                        for j in labels_list:
                            f.write(j)
                            f.write("\n")
                    img_path = os.path.join(save_dir, f"images/train_class{obj}/")
                    self.create_dir(img_path)
                    img_name = file.split(".")[0] + ".jpg"
                    shutil.copy(data_dir + f"images/train2017/{img_name}", img_path)


def main(args):

    train_data = CocoCustomizeExcept(args.data_dir, "labels/train2017")
    train_data.collect_data(
        args.data_dir, args.save_dir, args.extract_class
    )  # 56: chair, 60: dining table


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
        "--extract_class",
        help="extract list, 56: chair, 60: dining table",
        nargs="+",
        type=int,
        default=[56, 60],
    )

    args = parser.parse_args()
    main(args)
