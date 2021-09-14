from sklearn import metrics
import numpy as np
import re
from utils import performance_clustering, predict_martrix

#!!!!!!!!!!only for 360*10!!!!!!!!!!!!!!

#构建real矩阵b
a = np.ones((10,360))
for i in range(10):
    a[i] = (i+1)
b = a.reshape((1,3600))

c = predict_martrix("ml2d_images1.xmd", '000000                    '  ,17)[0]
acc, nmi = performance_clustering(b[0], c)
print("ml2d_正确率： " + str(acc))
print("ml2d_NMI: " + str(nmi))