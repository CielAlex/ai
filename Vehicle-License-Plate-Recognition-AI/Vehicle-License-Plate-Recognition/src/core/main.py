# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:25:53 2018

@author: Elliott
"""
import os
from Process import Data as data
from Feature import Feature as F
from Train import Train as Train
if __name__=='__main__':
    pic_info_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Char_Index.txt'
    pic_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Char_Image\\'
    grey_pic_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Gray_Image\\'
    pic_wrong_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Wrong_index.txt'
    feature_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\\'
    
    #图像二值化处理
    #process_data = data(pic_info_path,pic_path,grey_pic_path,pic_wrong_path)
    #process_data.preProcessing()
    #错误图像的处理
    #process_data.preProcessiongByHand()
    
    #特征提取
    #feature =F(pic_info_path,grey_pic_path,feature_path)
    #feature1
    #feature.Feature1Extraction1()
    #feature2
    #feature.Feature1Extraction2(3)
    #feature3
    #feature.Feature1Extraction3(3)
    
    #train
    T = Train(feature_path,pic_info_path)
    #T.write_data()
    T.train_data()

