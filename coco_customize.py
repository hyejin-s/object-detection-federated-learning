import os
from tqdm import tqdm
import pickle
import random
import matplotlib.pyplot as plt
import argparse

class CocoCustomize():
    def __init__(self, data_path, partition="train2017"):
        self.data_path = data_path
        self.dir = os.path.join(data_path, partition)
        self.file_list = os.listdir(self.dir)

    def read_file(self, file_name):
        '''
        Given .txt file, read the file line by line
        return : list of str ('class coordinate')
        '''
        f = open(os.path.join(self.dir, file_name), 'r')
        lines = f.readlines()
        lines = [line.strip() for line in lines]        
        f.close()
        
        return lines

    def class_removal(self, file_name, remove_list: list):
        '''
        Remove classes in input list
        return: list of str ('class coordinate')
        '''
        obj_list = self.read_file(file_name)
        obj_remov_list = obj_list.copy()

        for object in obj_list:
            if int(object.split()[0]) in remove_list:
                obj_remov_list.remove(object)

        return obj_remov_list

    def class_removal_proba(self, file_name, remove_list: list, probability: float):
        '''
        Remove classes probabilistically in input list (remaining the number of class: probability*100%)
        return: list of str ('class coordinate')
        '''
        obj_list = self.read_file(file_name)
        obj_remov_list = obj_list.copy()

        for obj in obj_list:
            if int(obj.split()[0]) in remove_list:
                a = random.randint(1, 1/probability)
                print(a)
                if a != 1:
                    obj_remov_list.remove(obj)

        return obj_remov_list

    def customize_data(self, remove_list: list, proba_num):
        '''
        Remove the class in total dataset.
        (Leave it with 1/proba_num probability)
        '''
        num = 0
        for file in tqdm(self.file_list):
            after_remove_list = self.class_removal(file, remove_list)
            if self.read_file(file) != after_remove_list: ### 클래스가 존재해서 지워진 것 중에
                num += 1
                if num % proba_num != 1: ### proba_num 에 해당하지 않으면 (지운 애들)
                    with open(os.path.join(self.dir, file), "w+") as f:
                        for obj in after_remove_list:
                            f.write(obj)
                            f.write("\n")

                    with open(f"./{proba_num}_remove_file/{file}", "a") as f:
                        for i in self.read_file(file):
                            f.write(i)
                            f.write("\n")

                    # print("success remove")
                else:
                    with open(f"{proba_num}_not_delete_file.txt", "a") as f:
                        f.write(file)
                        f.write("\n")
                    print("success save")

def main(args):

    train_data = CocoCustomize(args.data_path, "train2017")
    train_data.customize_data(args.remove_class, args.remove_num)
    file_list = train_data.file_list
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/hdd/hdd3/coco_custom/labels")
    parser.add_argument("--root", help="save root", type=str, default="/home/phj/object-detection-federated-learning/")
    parser.add_argument("--remove_num", help="remove ratio", type=int, default=100)
    parser.add_argument("--remove_class", help="remove list, 51: carrot, 60: dining table", default=[51, 60])
    args = parser.parse_args()

    main(args)
