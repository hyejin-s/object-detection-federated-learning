import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

import argparse

class DatasetDist:
    def __init__(self, data_path, partition="train2017"):
        self.data_path = data_path
        self.dir = os.path.join(data_path, partition)
        self.file_list = os.listdir(self.dir)

    def read_file(self, file_name):
        """
        Given .txt file, read the file line by line
        return : list of str ('class coordinate')
        """
        f = open(os.path.join(self.dir, file_name), "r")
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        f.close()

        return lines

    def img_class_list(self, file_name):
        """
        return: list of class per img
        """
        obj_list = self.read_file(file_name)
        obj_class_list = [int(line.split()[0]) for line in obj_list]

        return obj_class_list

    def count_class(self, save_name):
        """
        Count the number of classes in the entire file
        return: dictionary (key: class, value: the number of class)
        """
        class_dic = dict()
        for file in tqdm(self.file_list):
            img_class = self.img_class_list(file)
            for i in img_class:
                if i in class_dic:
                    class_dic[i] += 1
                else:
                    class_dic[i] = 1

        with open(save_name+".pickle", "wb") as f:
            pickle.dump(class_dic, f)

        return class_dic

def main(args):
    
    if args.dist:
        with open(args.name+".pickle", "rb") as f:
            class_dic = pickle.load(f)

        print(class_dic)

    else:
        dataset = DatasetDist(args.data_dir, args.name)
        dataset.count_class(args.name)

        with open(args.name+".pickle", "rb") as f:
            class_dic = pickle.load(f)

        print(class_dic)

        plt.figure(figsize=(15, 10))
        plt.bar(*zip(*class_dic.items()))
        plt.xlabel('Class', fontsize=20)
        plt.ylabel('The number of class', fontsize=20)
        plt.title('COCO train dataset class distribution', fontsize=25)
        plt.savefig(args.name+'.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/hdd/hdd3/coco_custom/labels")
    parser.add_argument("--name", help="folder name which wants to analyze", type=str, default="train2017")
    parser.add_argument("--dist", help="if pickle file is already saved and only prints", default=False)
    
    args = parser.parse_args()
    main(args)

    