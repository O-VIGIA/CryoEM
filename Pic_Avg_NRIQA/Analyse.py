import xlrd
import numpy as np
import tools as tl
from datetime import date,datetime

filepath = input("The filepath of Excel: ")

data = xlrd.open_workbook(filepath)
table = data.sheet_by_name(u'Sheet1')
print('---------------------BEGIN------------------------')
#获取行数和列数
nrows = table.nrows
ncols = table.ncols
print('Row: %d   Col: %d' % (nrows, ncols))

#记录图片名称
pic_name = table.col_values(0)[1:]
print(pic_name)

#记录得分到矩阵
martrix_data = [[] for i in range(nrows-1)]
# print(martrix_data)
for i in range(1, nrows):
        for j in range(1, ncols):
                martrix_data[i-1].append(round(float(table.cell(i, j).value),2))

martrix_data = np.array(martrix_data)
# print(martrix_data)
martrix_data_T = martrix_data.T
# print(martrix_data_T)
#对每个算子的得分分布进行标准化
martrix_data_norm = []
for i in range(ncols-1):
        martrix_data_norm.append(tl.standardization(martrix_data_T[i]))
martrix_data_norm = np.array(martrix_data_norm)
martrix_data_norm = martrix_data_norm.T
# print(martrix_data_norm)
#每行得分
print('--------------PicScore---------------')
score_sum = np.sum(martrix_data_norm, axis=1)
for i in range(nrows-1):
        print('Pic_name: %s   Pic_score: %f' % (pic_name[i], score_sum[i]))

# #按行求和-求出总得分值
# score_sum = np.sum(martrix_data,axis=1)
# print(score_sum)
# #标准化
# score_sum_norm = tl.standardization(score_sum)
# print(score_sum_norm)

#总得分（未加权可靠系数）
score_all = np.sum(score_sum)
score_all_fenbu = np.var(score_sum)
print('未加权聚类总得分：')
print(score_all)
print('差异系数： ')
print(score_all_fenbu)

#每个图像评价总得分的可靠性系数
# score_fenbu = [[0 for i in range(ncols-1)] for j in range(nrows-1)]
score_fenbu = [[] for i in range(nrows-1)]
num = 0
#对每一列得分进行从小到大排序
for i in range(ncols-1):
        temp_score_paixu = martrix_data_T[i].argsort()
        #第i个图像的得分排名
        for j in range(nrows-1):
                score_fenbu[temp_score_paixu[j]].append(j)
score_fenbu = np.array(score_fenbu)
print('---------------权重矩阵----------------')
print(score_fenbu)

#求可靠性系数（方差）排名的方差
relia_coefficient = np.var(score_fenbu, axis=1)
# print(relia_coefficient)
eoff_b = np.max(relia_coefficient) + np.min(relia_coefficient)
eoff_w = -1
adjust_relia_coefficient = eoff_w * relia_coefficient + eoff_b
print('---------------Pic_weight----------------')
for i in range(nrows-1):
        print('Pic_name: %s   Pic_weight: %f' % (pic_name[i], adjust_relia_coefficient[i]))

#总得分（加权 可靠系数）
score_sum_addweight = 0
for i in range(nrows-1):
        score_temp = score_sum[i]*adjust_relia_coefficient[i]
        # print(score_temp)
        score_sum_addweight += score_temp
print('加权得分：%f' % score_sum_addweight)

print('----------------------END-------------------------')

