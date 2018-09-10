# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:58:16 2018

@author: Elliott
"""
import numpy as np
import cv2 as cv
class Feature:
    def __init__(self,pic_info_path,pic_path,feature_path):
        self.pic_info_path =pic_info_path
        self.pic_path =pic_path
        self.feature_path=feature_path
    
    #每行 列白点数
    def Feature1Extraction1(self):
        data =np.loadtxt(self.pic_info_path,dtype =str,skiprows=1,delimiter='\t')
        index =data[:,0]
        datacategory =data[:,1]
        pic =data[:,2]
        length = pic.shape[0]
        filepath1 = self.feature_path+'feature1.txt'
        file =open(filepath1,'w+')
        for i in range(length):
            image =cv.imread(self.pic_path+pic[i],0)
            pic_length ,pic_width=image.shape  #92 47
            featurelist1=[]
            for lines in range(pic_length):
                count =0
                for columns in range(pic_width):
                    value  =image[lines,columns]
                    if (value==0):
                            count +=1
                file.writelines(str(count/pic_width))           
                file.writelines('\t')
            for lines in range(pic_width):
                count = 0
                for columns in range(pic_width):
                    value =image[lines,columns]
                    if (value==0):
                            count +=1
                file.writelines(str(count/pic_width))
                file.writelines('\t')
            file.writelines('\n')
        file.close()
    
    #区域密度
    def Feature1Extraction2(self,kernel_size):
        data =np.loadtxt(self.pic_info_path,dtype=str,skiprows=1,delimiter='\t')
        index  =data[:,0]
        datacategory =data[:,1]
        pic =data[:,2]
        legth =pic.shape[0]
        #image test
        #filepath_test =self.feature_path+'featuretest.txt'
        #filetest =open(filepath_test,'w+')
        #image =cv.imread(self.pic_path +pic[0],0)
        #pic_length ,pic_width=image.shape  #92 47
        #for i in range(pic_length):
        #    for j in range(pic_width):
        #        filetest.writelines(str(image[i,j]))
        #        filetest.writelines('\t')
        #    filetest.writelines('\n') 
        #filetest.close()
        
        filepath2 = self.feature_path+'feature2.txt'
        file =open(filepath2,'w+')
        for i in range(legth):
            image =cv.imread(self.pic_path+pic[i],0)
            pic_length ,pic_width=image.shape  #92 47
            kernel_line_size = int(pic_length/kernel_size)-1
            kernel_col_size = int(pic_width/kernel_size)-1
            for lines in range(kernel_line_size):
                for columns in range(kernel_col_size):
                    total =0
                    for parameter_line in range(kernel_size):
                        for parameter_column in range(kernel_size):
                            total += image[lines*kernel_size+parameter_line,columns*kernel_size+parameter_column]
                    file.writelines(str(total))
                    file.writelines('\t')
            file.writelines('\n')
        file.close()
        
    #图像区域取最大值代替原区域，模糊处理
    def Feature1Extraction3(self,kernel_size):
        data =np.loadtxt(self.pic_info_path,dtype=str,skiprows=1,delimiter='\t')
        index  =data[:,0]
        datacategory =data[:,1]
        pic =data[:,2]
        legth =pic.shape[0]
        
        filepath3 = self.feature_path+'feature3.txt'
        file =open(filepath3,'w+')
        for i in range(legth):
            image =cv.imread(self.pic_path+pic[i],0)
            pic_length ,pic_width=image.shape  #92 47
            kernel_line_size = int(pic_length/kernel_size)-1
            kernel_col_size = int(pic_width/kernel_size)-1
            for lines in range(kernel_line_size):
                for columns in range(kernel_col_size):
                    maxnum =0
                    for parameter_line in range(kernel_size):
                        for parameter_column in range(kernel_size):
                            if maxnum<image[lines*kernel_size+parameter_line,columns*kernel_size+parameter_column]:
                                maxnum =image[lines*kernel_size+parameter_line,columns*kernel_size+parameter_column]
                    file.writelines(str(maxnum))
                    file.writelines('\t')
            file.writelines('\n')
        file.close()        
        
    def Feature1Extraction4(self):
        pass
        
    def Feature1Extraction5(self):
        pass
        
    def Feature1Extraction6(self):
        pass
