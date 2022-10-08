import os
from tqdm import tqdm
import matplotlib.pyplot as plt


subfolder_list = list()
root = "./LGE-dataset"

for item in os.listdir(root): 
    sub_folder = os.path.join(root, item)
    if os.path.isdir(sub_folder):
        print(sub_folder)
        subfolder_list.append(sub_folder)

class_dict = dict()

for folder in subfolder_list:
    print(folder)
    for item in os.listdir(folder):
        file_path = os.path.join(folder, item)
        if file_path[-7:] == 'od2drgb':
        
            with open(file_path, encoding='utf8', errors='ignore') as f:
                contents = f.read()
                contents_read = contents[7:].split('\n')
                for i in range(len(contents_read)):
                    if i != 0 and i != len(contents_read)-1:
                        class_name = contents_read[i].split(', ')[-1]
                        if class_name in class_dict:
                            class_dict[class_name] += 1
                        else:
                            class_dict[class_name] = 1

print(class_dict)                

plt.figure(figsize=(25, 30))
plt.bar(*zip(*class_dict.items()))
plt.xlabel('Class', fontsize=25)
plt.ylabel('The number of class', fontsize=25)
plt.xticks(rotation=75, fontsize=20)
plt.yticks(fontsize=20)
plt.title('LGE dataset train dataset class distribution', fontsize=25)
plt.savefig('LGE_dataset_distribution.png')
plt.show()