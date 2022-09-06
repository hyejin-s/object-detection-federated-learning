import os
from tqdm import tqdm
import pickle
import random
import matplotlib.pyplot as plt
import argparse
import numpy as np
import shutil


class CocoCustomizeExcept:
    def __init__(self, data_path, partition="train2017"):
        self.data_path = data_path
        self.dir = os.path.join(data_path, partition)
        self.file_list = os.listdir(self.dir)

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

    def collect_data(self, collect_class_list: list):
        """
        Remove the class except collect_class_list in total dataset.
        Then, save that imgs and labels.
        """
        num = 0
        for file in tqdm(self.file_list):
            after_remove_list = self.class_removal(
                file, collect_class_list
            )  ### file 안에서 class 있는지 확인

            if len(after_remove_list) != 0:  ### 남아 있는 게 있으면
                obj_list = list()
                for i in range(len(after_remove_list)):
                    obj_class = int(after_remove_list[i].split()[0])
                    obj_list.append(obj_class)
                obj_list_uniq = np.unique(obj_list)  ### array([57, 58])

                """ save imgs and labels for each classes """
                for obj in obj_list_uniq:
                    labels_list = [
                        i for i in after_remove_list if int(i.split()[0]) == obj
                    ]
                    with open(
                        os.path.join(args.root, f"data/class{obj}/labels/{file}"), "w+"
                    ) as f:
                        for j in labels_list:
                            f.write(j)
                            f.write("\n")
                    img_path = os.path.join(args.root, f"data/class{obj}/images/")
                    img_name = file.split(".")[0] + ".jpg"
                    shutil.copy(
                        args.data_path + f"images/train2017/{img_name}", img_path
                    )

    def class_removal_proba(self, file_name, remove_list: list, probability: float):
        """
        Remove classes probabilistically in input list (remaining the number of class: probability*100%)
        return: list of str ('class coordinate')
        """
        obj_list = self.read_file(file_name)
        obj_remov_list = obj_list.copy()

        for obj in obj_list:
            if int(obj.split()[0]) in remove_list:
                a = random.randint(1, 1 / probability)
                print(a)
                if a != 1:
                    obj_remov_list.remove(obj)
        return obj_remov_list


def main(args):

    train_data = CocoCustomizeExcept(args.data_path, "labels/train2017")
    file_list = train_data.file_list
    train_data.collect_data([51, 60])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        help="save root",
        type=str,
        default="/home/phj/object-detection-federated-learning/",
    )
    parser.add_argument("--data_path", type=str, default="/hdd/hdd3/coco/")
    args = parser.parse_args()

    main(args)
