# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:27:58 2018
@author: Elliott
"""
import os
import numpy as np
import cv2 as cv
class Data:
    def __init__(self,pic_info_path,pic_path,grey_pic_path,pic_wrong_path):
        self.pic_info_path = pic_info_path
        self.pic_path =pic_path
        self.grey_pic_path =grey_pic_path
        self.pic_wrong_path =pic_wrong_path
    
    def preProcessing(self):
        if(os.path.exists(self.pic_info_path)):
            data = np.loadtxt(self.pic_info_path ,dtype =str ,skiprows=1,delimiter='\t')
            index = data[:,0]
            category = data[:,1]
            pic = data[:,2]
            for i in range(1000):
                #图像的二值化
                image =cv.imread(self.pic_path+pic[i])
                gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
                ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
                #cv.namedWindow("binary0", cv.WINDOW_NORMAL)
                #cv.imshow("binary0", binary)
                length ,width = binary.shape
                sum =0
                for j in range(4):
                    sum += binary[j,j]                    
                    sum += binary[j,j+width-4]
                    sum += binary[length-4+j,j]
                    sum += binary[length-4+j,width-4+j]
                if sum >(8*255):
                    #二次二值化（针对部分二值化错误图像优化）
                    binary_new = cv.threshold(binary,0,255,cv.THRESH_BINARY_INV)
                    cv.imwrite(self.grey_pic_path+pic[i],binary_new[1])
                else:
                    #一般图像的二值化
                    cv.imwrite(self.grey_pic_path+pic[i],binary)        
        else:
            return False
    
    def preProcessiongByHand(self):
        if(os.path.exists(self.pic_info_path)):
            data = np.loadtxt(self.pic_wrong_path ,dtype =str ,skiprows=1)
            wrong_pic = data[:,1]
            length ,width =data.shape
            for i in range(length):
                image =cv.imread(self.grey_pic_path+wrong_pic[i],0)
                binary_new = cv.threshold(image,0,255,cv.THRESH_BINARY_INV)
                cv.imwrite(self.grey_pic_path+wrong_pic[i],binary_new[1])
            
        else:
            return False
