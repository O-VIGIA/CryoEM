from sklearn import metrics
import numpy as np
import re
from utils import performance_clustering, predict_martrix

#构建real矩阵b
a = np.ones((10,360))
for i in range(10):
    a[i] = (i+1)
b = a.reshape((1,3600))

zimulu1 = ['5001', '6840', '11669']
zimulu2 = ['01', '03', '05']
acc = 0
nmi = 0

for name1 in zimulu1:

    for name2 in zimulu2:
        acc = 0
        nmi = 0
        for i in range(3):

            j = i + 1
            wenjian_path = "E://count_accuracy//ML2D//" + name1 + "//" + name2 + "//" + "ml2d_images" + str(j) +".xmd"
            c = predict_martrix(wenjian_path, '000000                    ', 17)[0]
            a, n = performance_clustering(b[0], c)
            print("EMD-"+name1+"    信噪比"+name2+"   实验次数"+str(j))
            print("acc: %f" % a)
            print("nmi: %f" % n)
            acc = acc + a
            nmi = nmi + n
        print("---------------------AVG_BEGIN--------------------")
        print("EMD-"+name1+"    信噪比"+name2+"  AVG  ")
        print("acc_avg: " + str(acc/3))
        print("nmi_avg: " + str(nmi/3))
        print("---------------------AVG_ENDED--------------------")
