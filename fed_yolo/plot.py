## temporary

import matplotlib.pyplot as plt

def read_file(file_name):
        """
        Given .txt file, read the file line by line
        return : list of str ('class coordinate')
        """
        f = open(file_name, "r")
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        f.close()

        return lines

results = read_file('./output/exp3/server.txt')
results_coco = read_file('./output/exp4/server.txt')
mAP, mAP_coco = list(), list()

for i, result in enumerate(results):
        if i != 0:
                mAP.append(float(result.split(' ')[3][:-1]))

for i, result in enumerate(results_coco):
        if i != 0:
                mAP_coco.append(float(result.split(' ')[3][:-1]))
# print(mAP_coco)
# print(len(mAP_coco))
        
plt.figure(figsize=(15, 10))
X = list(range(1,100+1))
plt.plot(X, mAP[:100], label='train with server', lw=2)
plt.plot(X, mAP_coco[:100], label='class56, 60', lw=2)
plt.xlabel('Round', fontsize=20)
plt.ylabel('mAP', fontsize=20)
plt.legend(fontsize=20)
# plt.title('COCO train dataset class distribution', fontsize=25)
plt.savefig('test.png')
# plt.savefig('coco_train_distribution_custom.pdf')
plt.show()


# def plotRegret(filepath):
#     plt.figure()
#     X = list(range(1, M+1))
#     for i in range(len(POLICIES)):
#         plt.plot(X, regret_transfer[i], label="{}".format(labels[i]), color=colors[i])
#     plt.legend()
#     plt.title("Total {} episode, {} horizon".format(M, HORIZON))
#     plt.savefig(filepath+'/Regret', dpi=300)