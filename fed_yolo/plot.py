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


results_server = read_file("./output/exp6/server.txt")
results_local = read_file("./output/exp17/server.txt")

mAP_s, mAP_l = list(), list()

for i, result in enumerate(results_server):
    if i != 0:
        mAP_s.append(float(result.split(" ")[3][:-1]))

for i, result in enumerate(results_local):
    if i != 0:
        mAP_l.append(float(result.split(" ")[3][:-1]))

results_s2 = read_file("./output/exp18/server.txt")
results_s2l10 = read_file("./output/exp19/server.txt")
results_l10 = read_file("./output/exp20/server.txt")

mAP, mAP2, mAP3 = list(), list(), list()

for i, result in enumerate(results_s2):
    if i != 0:
        mAP.append(float(result.split(" ")[3][:-1]))

for i, result in enumerate(results_s2l10):
    if i != 0:
        mAP2.append(float(result.split(" ")[3][:-1]))

for i, result in enumerate(results_l10):
    if i != 0:
        mAP3.append(float(result.split(" ")[3][:-1]))

# print(mAP_coco)
# print(len(mAP_coco))

plt.figure(figsize=(15, 10))
X = list(range(1, len(mAP_s) + 1))
# plt.plot(X, mAP_l[:len(mAP_s)], label="training with local epoch 1", lw=2)

# plt.plot(X, mAP_s[:len(mAP_s)], label="training with server epoch 1", lw=2)
plt.plot(
    X,
    mAP[: len(mAP_s)],
    label="training with server epoch 2 and local epoch 1",
    lw=2,
    color="b",
)
plt.plot(
    X,
    mAP2[: len(mAP_s)],
    label="training with server epoch 2 and local epoch 10",
    lw=2,
    color="r",
)
plt.plot(X, mAP3[: len(mAP_s)], label="training only local epoch 10", lw=2, color="g")


plt.xlabel("Round", fontsize=25)
plt.ylabel("mAP", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(fontsize=20)
# plt.title('COCO train dataset class distribution', fontsize=25)
plt.savefig("epoch_compare.png")
plt.show()


# -----------------
results_1_16 = [float(i) for i in read_file("./output/exp16/server_class_56.txt")]
results_1_17 = [float(i) for i in read_file("./output/exp17/server_class_56.txt")]
results_1_18 = [float(i) for i in read_file("./output/exp18/server_class_56.txt")]
results_1_19 = [float(i) for i in read_file("./output/exp19/server_class_56.txt")]
results_1_20 = [float(i) for i in read_file("./output/exp20/server_class_56.txt")]
# print(float(results_1_16))
plt.figure(figsize=(15, 10))
X = list(range(1, 10 + 1))
# plt.plot(X, results_1_16[:10], label="16", lw=2)
# plt.plot(X, results_1_17[:10], label="17", lw=2, color='g')

plt.plot(
    X,
    results_1_18[:10],
    label="training with server epoch 2 and local epoch 1",
    lw=2,
    color="r",
)
plt.plot(
    X,
    results_1_19[:10],
    label="training with server epoch 2 and local epoch 10",
    lw=2,
    color="b",
)
plt.plot(X, results_1_20[:10], label="training only local epoch 10", lw=2, color="g")

plt.xlabel("Round", fontsize=25)
plt.ylabel("mAP", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(fontsize=20)
# # plt.title('COCO train dataset class distribution', fontsize=25)
plt.savefig("class56_server.png")
plt.show()

results_1_16 = [float(i) for i in read_file("./output/exp16/server_class_60.txt")]
results_1_17 = [float(i) for i in read_file("./output/exp17/server_class_60.txt")]
results_1_18 = [float(i) for i in read_file("./output/exp18/server_class_60.txt")]
results_1_19 = [float(i) for i in read_file("./output/exp19/server_class_60.txt")]
results_1_20 = [float(i) for i in read_file("./output/exp20/server_class_60.txt")]
# print(float(results_1_16))
plt.figure(figsize=(15, 10))
X = list(range(1, 10 + 1))
# plt.plot(X, results_1_16[:10], label="16", lw=2)
# plt.plot(X, results_1_17[:10], label="17", lw=2, color='g')

plt.plot(
    X,
    results_1_18[:10],
    label="training with server epoch 2 and local epoch 1",
    lw=2,
    color="r",
)
plt.plot(
    X,
    results_1_19[:10],
    label="training with server epoch 2 and local epoch 10",
    lw=2,
    color="b",
)
plt.plot(X, results_1_20[:10], label="training only local epoch 10", lw=2, color="g")

plt.xlabel("Round", fontsize=25)
plt.ylabel("mAP", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(fontsize=20)
# # plt.title('COCO train dataset class distribution', fontsize=25)
plt.savefig("class60_server.png")
plt.show()
