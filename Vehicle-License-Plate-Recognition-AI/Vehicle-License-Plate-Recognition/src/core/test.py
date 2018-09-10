# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:58:16 2018

@author: Elliott
"""
from libsvm.python.svm import *
from libsvm.python.svm import __all__ as svm_all
import os 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Char_Index.txt'
index =[]
category =[]
pic =[]
file_path = ''
if __name__=='__main__':
    a = np.array([[1,2,3]]);
    b = np.array([[4,5,6]]);
    c =np.concatenate((a,b))
    
    
    if(os.path.exists(path)):
        data = np.loadtxt(path ,dtype =str ,skiprows=1,delimiter='\t')
        index = data[:,0]
        category = data[:,1]
        pic = data[:,2]
        for i in range(1000):
            #图像的二值化
            image =cv.imread('F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Char_Image\\'+pic[i])
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
            ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
            #cv.namedWindow("binary0", cv.WINDOW_NORMAL)
            #cv.imshow("binary0", binary)
            cv.imshow("binary1",binary)
            ret, binary1 = cv.threshold(binary,0,255,cv.THRESH_BINARY_INV)
            cv.imshow("binary2",binary1)
            cv.imwrite('F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Gray_Image\\'+pic[i],binary)
        
        pass
    else:
        pass
            