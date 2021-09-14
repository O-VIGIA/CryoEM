import numpy as np
from sklearn import metrics
from munkres import Munkres
#acc, nmi = mrcdata_process.performance_clustering(labels_true, labels_predict)

def performance_clustering(labels_real, labels_predict):
    label_same = best_map(labels_real, labels_predict)
    count = np.sum(labels_real[:] == label_same[:])
    acc = count.astype(float) / (labels_real.shape[0])
    nmi = metrics.normalized_mutual_info_score(labels_real, label_same)
    return acc, nmi

def best_map(L1, L2):
    # L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)  # 去除重复的元素，由小到大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def predict_martrix(path, substr, index):
    c = np.ones((1, 3600))
    k = 0
    with open(path, "r") as f:
        lines = f.readlines()
    add_num = len(substr)
    for line in lines[index:]:
        pre_lei = line.find(substr) + add_num
        # print(pre_lei)
        c[0][k] = int(line[pre_lei])
        k = k + 1
        # print(c[0][k])
    f.close()
    return c
