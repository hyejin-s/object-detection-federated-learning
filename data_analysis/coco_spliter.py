import os
from tqdm import tqdm
import random
import argparse
import shutil


class CocoSpliter:
    class1_count = 0
    class2_count = 0
    class1n2_count = 0

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

    def check_file_has_class(self, file_name, class_list: list):
        assert len(class_list) == 2, "extract class list len should be 2"
        file_obj_list = self.read_file(file_name)
        [class1, class2] = class_list
        has_class1 = any([int(obj.split()[0]) == class1 for obj in file_obj_list])
        has_class2 = any([int(obj.split()[0]) == class2 for obj in file_obj_list])
        if has_class1 and has_class2:
            self.class1n2_count += 1
            return class1 if random.random() < 0.7 else class2
        elif has_class1 and not has_class2:
            self.class1_count += 1
            return class1
        elif not has_class1 and has_class2:
            self.class2_count += 1
            return class2
        else:
            return (
                class1
                if random.random() < 0.02
                else class2
                if random.random() < 0.02
                else -1
            )

    def split_data(self, data_dir, save_dir, collect_class_list: list):
        """
        Split the dataset into class1, class2, remains.
        """
        assert len(collect_class_list) == 2, "extract class list len should be  2"
        for file in tqdm(self.file_list):  ## labels
            obj = self.check_file_has_class(file, collect_class_list)
            if obj == -1:
                label_path = os.path.join(save_dir, f"labels/server/")
                self.create_dir(label_path)
                shutil.copy(f"{self.dir}/{file}", label_path)
                
                img_path = os.path.join(save_dir, f"images/server/")
                self.create_dir(img_path)
                img_name = file.split(".")[0] + ".jpg"
                shutil.copy(f"{data_dir}images/train2017/{img_name}", img_path)
                
            else: # for node
                if obj == collect_class_list[0]:
                    node = 1
                else:
                    node = 2
                label_path = os.path.join(save_dir, f"labels/node_{node}_class_{obj}/")
                self.create_dir(label_path)
                shutil.copy(f"{self.dir}/{file}", label_path)
                img_path = os.path.join(save_dir, f"images/node_{node}_class_{obj}/")
                self.create_dir(img_path)
                img_name = file.split(".")[0] + ".jpg"
                shutil.copy(f"{data_dir}images/train2017/{img_name}", img_path)
        print(
            f"class1:{self.class1_count}\nclass2:{self.class2_count}\nclass1&2:{self.class1n2_count}"
        )


def main(args):

    train_data = CocoSpliter(args.data_dir, "labels/train2017")
    train_data.split_data(
        args.data_dir, args.save_dir, args.extract_class
    )  # 56: chair, 60: dining table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        help="save root",
        type=str,
        default="/hdd/hdd3/coco_fl/",
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
