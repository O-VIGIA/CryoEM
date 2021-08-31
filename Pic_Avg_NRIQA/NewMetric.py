import cv2
import matplotlib.pyplot as plt
import numpy as np
import math as mth
import os as osc
import xlsxwriter
from skimage.feature import greycomatrix, greycoprops
import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi
import iBin_Enc_Dec as ed
import scipy.stats as ss

#二维熵
def calc_2D_Entropy(img, N = 1):
    a = [i for i in range(256)]
    S = img.shape
    IJ = []
    for row in range(S[0]):
        for col in range(S[1]):
            left_x = np.max([0, col-N])
            right_x = np.min([S[1], col+N+1])
            up_y = np.max([0, row-N])
            down_y = np.min([S[0], row+N+1])
            region = img[up_y:down_y, left_x:right_x]
            j = (np.sum(region) - img[row][col])/((2*N+1)**2-1)
            IJ.append([img[row, col], j])
    #print(IJ)
    F = []
    arr = [list(i) for i in set(tuple(j) for j in IJ)]
    for i in range(len(arr)):
        F.append(IJ.count(arr[i]))
    #print(F)
    P = np.array(F)/len(F)
    E = np.sum([p * np.log2(1/p) for p in P])
    # print("2D_Entropy_score:" + str(E))
    if E < 0 :
        E = 0
    return E

#SMD
def SMD(img):
    shape = img.shape
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(1, shape[1]):
            out += mth.fabs(img[x,y] - img[x,y-1])
            out += mth.fabs(img[x,y] - img[x+1,y])
    # print("SMD_Score:" + str(out))
    return out

#SMD2
def SMD2(img):
    shape = img.shape
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out += mth.fabs(img[x,y]-img[x+1,y]) * mth.fabs(img[x,y]-img[x,y+1])
    return out

#Energy
def Energy(img):
    shape = img.shape
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out += (img[x+1,y]-img[x,y])**2 + (img[x,y+1]-img[x,y])**2
    return out

#Vollath函数
def Vollath(img):
    shape = img.shape
    u = np.mean(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out += img[x,y]*img[x+1,y]
    out -= shape[0]*shape[1]*(u**2)
    return mth.fabs(out)

#EAV点锐度算法函数!!!
def EVA(image, size=1):
    '''
    扩展核的大小 距离度量
    '''
    corner = 1 / 2**0.5
    edge = 1.0
    Kernel_EVA = np.array([[corner, edge, corner],
                  [edge, -8, edge],
                  [corner, edge, corner]])
    dst = cv2.filter2D(img, cv2.CV_64F, Kernel_EVA)
    Score_EVA = np.mean(dst)
    return mth.fabs(Score_EVA)

#灰度共生矩阵
#角二阶矩阵
# gray_level = 16
# def maxGrayLevel(img):
#     max_gray_level = 0
#     (height, width) = img.shape
#     for x in range(height):
#         for y in range(width):
#             if img[x][y] > max_gray_level:
#                 max_gray_level = img[x][y]
#     return max_gray_level+1
#
# def getClcm(img):
#     #减小灰度级
#     values_temp = []
#     dst_data = img.copy()
#     max_gray_level = maxGrayLevel(img)
#     (height, width) = img.shape
#     for j in range(height):
#         for i in range(width):
#             dst_data[j][i] = dst_data[j][i]*gray_level / max_gray_level
#     #计算共生矩阵
#     glcm = greycomatrix(dst_data, [2, 8, 16], [0, np.pi/4, np.pi/2, np.pi*3/4], gray_level, symmetric=True, normed=True)
#     #计算表示特征的纹理参数
#     props = {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}
#     for prop in props:
#         temp = greycoprops(glcm, prop)
#         values_temp.append(temp)
#         print(prop, temp)
#     return values_temp



#NR NRSS
def compute_ssim(img_mat_1, img_mat_2):
    # Variables for Gaussian kernel definition
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))
    # Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = \
                (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) * \
                exp(-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))
    # Convert image matrices to double precision (like in the Matlab version)
    img_mat_1 = img_mat_1.astype(np.float)
    img_mat_2 = img_mat_2.astype(np.float)
    # Squares of input matrices
    img_mat_1_sq = img_mat_1 ** 2
    img_mat_2_sq = img_mat_2 ** 2
    img_mat_12 = img_mat_1 * img_mat_2
    # Means obtained by Gaussian filtering of inputs
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)
    # Squares of means
    img_mat_mu_1_sq = img_mat_mu_1 ** 2
    img_mat_mu_2_sq = img_mat_mu_2 ** 2
    img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2
    # Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)
    # Covariance
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)
    # Centered squares of variances
    img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
    img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12;
    # c1/c2 constants
    # First use: manual fitting
    c_1 = 6.5025
    c_2 = 58.5225
    # Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l = 255
    k_1 = 0.01
    c_1 = (k_1 * l) ** 2
    k_2 = 0.03
    c_2 = (k_2 * l) ** 2
    # Numerator of SSIM
    num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
    # Denominator of SSIM
    den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
               (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
    # SSIM
    ssim_map = num_ssim / den_ssim
    index = np.average(ssim_map)
    # print(index)
    return index

def gauseBlur(img):
    img_Guassian = cv2.GaussianBlur(img,(7,7),0)
    return img_Guassian

def loadImage(filepath):
    img = cv2.imread(filepath, 0)  ##   读入灰度图
    return img
def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
def saveImage(path, img):
    cv2.imwrite(path, img)
def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst
def getBlock(G,Gr):
    (h, w) = G.shape
    G_blk_list = []
    Gr_blk_list = []
    sp = 6
    for i in range(sp):
        for j in range(sp):
            G_blk = G[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
            Gr_blk = Gr[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
            G_blk_list.append(G_blk)
            Gr_blk_list.append(Gr_blk)
    sum = 0
    for i in range(sp*sp):
        mssim = compute_ssim(G_blk_list[i], Gr_blk_list[i])
        sum = mssim + sum
    nrss = 1-sum/(sp*sp*1.0)
    return nrss
def NRSS(image):
    #高斯滤波
    Ir = gauseBlur(image)
    G = sobel(image)
    Gr = sobel(Ir)
    blocksize = 8
    ## 获取块信息
    Nrss = getBlock(G, Gr)
    return Nrss


#超参数
std_luminance_quant_tbl = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])
# Scan reorder type for coefficients according to frequency
scan_order_zz=np.array([0,1,8,16,9,2,3,10,
                        17,24,32,25,18,11,4,5,
                        12,19,26,33,40,48,41,34,
                        27,20,13,6,7,14,21,28,
                        35,42,49,56,57,50,43,36,
                        29,22,15,23,30,37,44,51,
                        58,59,52,45,38,31,39,46,
                        53,60,61,54,47,55,62,63])
scan_order_row=np.array([0,1,2,3,4,5,6,7,
                         8,9,10,11,12,13,14,15,
                         16,17,18,19,20,21,22,23,
                         24,25,26,27,28,29,30,31,
                         32,33,34,35,36,37,38,39,
                         40,41,42,43,44,45,46,47,
                         48,49,50,51,52,53,54,55,
                         56,57,58,59,60,61,62,63])
L_kernel=np.array([[1.0,4.0,1.0],[4.0,-20.0,4.0],[1.0,4.0,1.0]])/6.0
Robert_operator_Horizontal=np.array([0,0,0, 0,1,0, -1,0,0]).reshape(3,3)
Robert_operator_Vertical=np.array([0,0,0,0,1,0,0,0,-1]).reshape(3,3)
Prewitt_operator_Horizontal=np.array([-1,0,1, -1,0,1, -1,0,1]).reshape(3,3)
Prewitt_operator_Vertical=np.array([-1,-1,-1, 0,0,0, 1,1,1]).reshape(3,3)
Horizontal_Sobel_3=np.array([1,2,1,-2,-4,-2,1,2,1]).reshape(3,3)
Vertical_Sobel_3=np.array([1,-2,1, 2,-4,2, 1,-2,1]).reshape(3,3)
blksize = 8

def Kur_NoDC(img):
    # Calculate DCT zeros
    DCTcoeffs = ed.DoDCTTrans(img, blksize)
    DCTCBlock = ed.DoZigZagScan(DCTcoeffs, blksize, scan_order_row)  # get result after zig zag scan
    # Kurtosis Calculation No DC components
    kurtNoDC = np.mean(ss.kurtosis(DCTCBlock[1:, :], axis=0, fisher=False, bias=False))
    return kurtNoDC

def Mean_of_Img(img):
    # mean of image
    val_mean = np.mean(img)
    return val_mean

def Var_of_Img(img):
    # Variance of intensity values
    val_var = np.var(img)
    return val_var

def Relative_Var(img):
    m = Mean_of_Img(img)
    v = Var_of_Img(img)
    return v/m

def Contrast(img):
    Contr=np.power((np.float32(img.max())-np.float32(img.mean())),2)/np.power(np.float32(img.max())+np.float32(img.mean()),2)
    return Contr

def T_sobel(img):
    iHeight, iWidth = img.shape
    # sobel
    xgrad2 = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    ygrad2 = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    Sim2 = (np.power(xgrad2, 2) + np.power(ygrad2, 2)) / 16
    # Tenengrad Sobel
    T_Sobel_S = np.sum(Sim2) / (iHeight * iWidth)
    return T_Sobel_S

def T_scharr(img):
    iHeight, iWidth = img.shape
    # scharr
    xgrad1 = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    ygrad1 = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    Sim1 = (np.power(xgrad1, 2) + np.power(ygrad1, 2)) / 16
    # Tenengrad Scharr
    T_Scharr_S = np.sum(Sim1) / (iHeight * iWidth)
    return T_Scharr_S

def cal_1D_Entropy(img):
    hst, bins = np.histogram(img.ravel(), range(256))
    max_freq = np.amax(hst)
    E = ss.entropy(hst / float(max_freq), qk=None, base=2.0)
    return E

def Squared_hor(img):
    iHeight, iWidth = img.shape
    SG_horizontal = np.sum(np.power(img[:, 1:] - img[:, 0:-1], 2))
    SG_horizontal = SG_horizontal / (iHeight * (iWidth - 1))
    return SG_horizontal

def Squared_ver(img):
    iHeight, iWidth = img.shape
    SG_vertical = np.sum(np.power(img[0:-1, :] - img[1:, :], 2))
    SG_vertical = SG_vertical / ((iHeight - 1) * iWidth)
    return SG_vertical

def Robert_G(img):
    iHeight, iWidth = img.shape
    # Apply Robert filter
    S_Robert_x = cv2.filter2D(img, cv2.CV_64F, Robert_operator_Horizontal)
    S_Robert_y = cv2.filter2D(img, cv2.CV_64F, Robert_operator_Vertical)
    G_Robert = (np.power(S_Robert_x, 2) + np.power(S_Robert_y, 2)) / 16
    Robert_S=np.sum(G_Robert)/(iHeight*iWidth)
    return Robert_S

def Prewitt_G(img):
    iHeight, iWidth = img.shape
    # Apply Prewitt
    S_Prewitt_x = cv2.filter2D(img, cv2.CV_64F, Prewitt_operator_Horizontal)
    S_Prewitt_y = cv2.filter2D(img, cv2.CV_64F, Prewitt_operator_Vertical)
    G_Prewitt = (np.power(S_Prewitt_x, 2) + np.power(S_Prewitt_y, 2)) / 16
    Prewitt_S=np.sum(G_Prewitt)/(iHeight*iWidth)
    return Prewitt_S

def Auto_cor(img):
    iHeight, iWidth = img.shape
    val_var = np.var(img)
    val_mean = np.mean(img)
    # Standard autocorrelation
    k = 2
    mat_1 = img[:-1, :-(k + 1)] - np.ones((iHeight - 1, iWidth - (k + 1)), dtype=np.float) * val_mean
    mat_2 = img[:-1, k:-1] - np.ones((iHeight - 1, iWidth - (k + 1)), dtype=np.float) * val_mean
    Auto_Correlation = (iHeight * iWidth - k) * val_var - np.sum(mat_1 * mat_2)
    return Auto_Correlation

def Sobel_5(img):
    iHeight, iWidth = img.shape
    # Apply Sobel filter kernel size 5
    sobel5_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # (src, dst, xorder, yorder, apertureSize=3)
    sobel5_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    S_sobel5 = (np.power(sobel5_x, 2) + np.power(sobel5_y, 2)) / 16
    SG_kernel5 = np.sum(S_sobel5) / (iHeight * iWidth)
    return SG_kernel5

def Mendelsohn_and_Mayall(img):
    Lim1 = cv2.filter2D(img, cv2.CV_64F, L_kernel)
    # Laplacian with external kernel mean
    L1m = Lim1.mean()
    # abs Laplacian with external kernel mean
    L1am = np.abs(Lim1).mean()
    # draw histogram分布直方图 ravel 扁平化函数 先统计（0，256）区间像素的分布 每个区间相应像素值的数量 相应区间
    hst, bins = np.histogram(img.ravel(), range(256))
    # Mendelsohn and Mayall
    mean_int = int(L1am)
    qty = 255 - mean_int + 1
    k = range(qty) + np.ones(qty, dtype=np.int) * mean_int
    hk = hst[mean_int - 1:]
    MenMay = sum(k * hk)
    return MenMay


Files_Path=r'D:\Cluana\code\CL1\clustering_analyze\imgs\pic_test\\'
print("----------------------Begin------------------------------")
fn = input("Filename: ")
Files_Path = Files_Path + fn + '\\'
onlyfiles = [f for f in osc.listdir(Files_Path) if osc.path.isfile(osc.path.join(Files_Path, f))]
Files_Path_Results=r'D:\Cluana\code\CL1\clustering_analyze\imgs\result_tets\\'
workbook = xlsxwriter.Workbook(Files_Path_Results+'New_Metrics_'+ fn +'.xlsx')
worksheet = workbook.add_worksheet()
Header = (['Number', 'Kurtosis-NoDC', 'Mean of Image', 'Var of Image', 'Relative Var', 'Contrast', 'T-sobel', 'T-scharr', 'Entropy', '2D_Entropy',
           'SQ_hor', 'SQ_ver', 'Robert_G', 'Prewitt_G', 'Auto_Cor', 'SMD', 'SMD2', 'EVA', 'NRSS', 'Voll', 'Sobel_5', 'Mendelsohn and Mayall'])
# Write Header to
rowx = 0
# length=np.size(Header)
# Iterate over the data and write it out row by row.
for col in range(0,np.size(Header)):
    worksheet.write(rowx, col, Header[col])
    col += 1
rowx=1


num = -1
for f in onlyfiles:
    num = num + 1
    fname = Files_Path + f
    print(fname)
    img = cv2.imread(fname, 0)
    Score_1D_entropy = cal_1D_Entropy(img)
    Score_2D_Entropy = calc_2D_Entropy(img)
    Score_SMD = SMD(img)
    Score_SMD2 = SMD2(img)
    Score_Energy = Energy(img)
    Score_Voll = Vollath(img)
    Score_EVA = EVA(img)
    Score_NRSS = NRSS(img)
    Score_KNDC = Kur_NoDC(img)
    Score_Mean = Mean_of_Img(img)
    Score_Var = Var_of_Img(img)
    Score_Relative_Var = Relative_Var(img)
    Score_Con = Contrast(img)
    Score_T_sobel = T_sobel(img)
    Score_T_scharr = T_scharr(img)
    Score_SQ_hor = Squared_hor(img)
    Score_SQ_ver = Squared_ver(img)
    Score_Robert = Robert_G(img)
    Score_Prewitt = Prewitt_G(img)
    Score_Auto_Cor = Auto_cor(img)
    Score_Sobel_5 = Sobel_5(img)
    Score_MM = Mendelsohn_and_Mayall(img)

    #write excel
    worksheet.write(rowx, 0, f)
    worksheet.write(rowx, 1, Score_KNDC)
    worksheet.write(rowx, 2, Score_Mean)
    worksheet.write(rowx, 3, Score_Var)
    worksheet.write(rowx, 4, Score_Relative_Var)
    worksheet.write(rowx, 5, Score_Con)
    worksheet.write(rowx, 6, Score_T_sobel)
    worksheet.write(rowx, 7, Score_T_scharr)
    worksheet.write(rowx, 8, Score_1D_entropy)
    worksheet.write(rowx, 9, Score_2D_Entropy)
    worksheet.write(rowx, 10, Score_SQ_hor)
    worksheet.write(rowx, 11, Score_SQ_ver)
    worksheet.write(rowx, 12, Score_Robert)
    worksheet.write(rowx, 13, Score_Prewitt)
    worksheet.write(rowx, 14, Score_Auto_Cor)
    worksheet.write(rowx, 15, Score_SMD / (10 ** 6))
    worksheet.write(rowx, 16, Score_SMD2 / (10 ** 8))
    worksheet.write(rowx, 17, Score_EVA)
    worksheet.write(rowx, 18, Score_NRSS*(10**2))
    worksheet.write(rowx, 19, Score_Voll / (10 ** 8))
    worksheet.write(rowx, 20, Score_Sobel_5)
    worksheet.write(rowx, 21, Score_MM)
    rowx+=1
workbook.close()
