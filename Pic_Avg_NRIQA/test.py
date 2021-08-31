import cv2
import numpy as np

GS_x = cv2.getGaussianKernel(3, 0.8)
GS_y = cv2.getGaussianKernel(3, 0.8).T

print(GS_x)
print(GS_y)

GS = np.dot(GS_x, GS_y)
print(GS)

def ChanshengGauss(size, σ):

    pass
