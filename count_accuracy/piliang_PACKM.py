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

count_num = 0
# accs=[]
# nmis=[]
wenjianjia = ['01', '03', '05']
print("PCA+KM 5次实验平均值")

for k in wenjianjia:
    acc = 0
    nmi = 0
    for i in range(4):
        j = i + 1
        shuju_path = 'E://count_accuracy//PAC_KM//' + k + '//docassign' + str(j) + '.stk'
        d = predict_martrix(shuju_path, "0       ", 2)[0]
        a, n = performance_clustering(b[0], d)
        print("信噪比"+k+"第"+str(j)+"次实验数据")
        print("acc: "+str(a))
        print("nmi: "+str(n))
        acc = acc + a
        nmi = nmi + n
    # accs[count_num] = acc / 5
    # nmis[count_num] = nmi / 5
    print("--------------AVG BEGIN------------------")
    print("信噪比" + k)
    print("acc: " + str(acc/4))
    print("nmi: " + str(nmi/4))
    count_num = count_num + 1
    print("---------------AVG END--------------------")
#c = predict_martrix("ml2d_images.xmd", '000000                    '  ,17)[0]
# acc, nmi = performance_clustering(b[0], c)
# print("ml2d_正确率： " + str(acc))
# print("ml2d_NMI: " + str(nmi))
#
# d = predict_martrix("docassign.stk", "0       ", 2)[0]
# acc, nmi = performance_clustering(b[0], d)
# print("docassign_正确率： " + str(acc))
# print("docassign_NMI: " + str(nmi))



# beishu = 360
# index = 17
# error_num = 0
# error_sum = 0
#
# for k in range(10):
#     #xiangmu
#     i = lines[index].find('projections_')
#     j = lines[index].find('.', 72)
#     tupian_lei = lines[index][i:j]
#     print(tupian_lei)
#
#     for line in lines[index:index+360]:
#         index_lei = line.find('000000                    ')
#         if(line[index_lei+num] != classes[k]) :
#             error_num = error_num + 1
#
#     error_ = error_num/360
#     print("错误率： " + str(error_))
#     print("正确率： " + str((1-error_)))
#     index = index+360
#     error_num = 0
#     error_sum += error_
# error_sum = error_sum/10
# print("总错误率： " + str(error_sum))
# print("总正确率: " + str(1-error_sum))







