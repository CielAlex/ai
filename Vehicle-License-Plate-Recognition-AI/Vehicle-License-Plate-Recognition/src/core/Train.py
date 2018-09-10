# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 01:16:12 2018

@author: Elliott
"""
from libsvm.python.svm import *
from libsvm.python.svm import __all__ as svm_all
from libsvm.python.svmutil import *
from libsvm.python.svmutil import __all__ as svmutil_all
import numpy as np
class Train:
    def __init__(self,feature_path,pic_info_path):
        self.feature_list = ['feature1.txt']
        self.feature_path = feature_path
        self.pic_info_path =pic_info_path
        
    def write_data(self):
        feature1 = np.loadtxt(self.feature_path+'feature1.txt',dtype=str,delimiter='\t')
        feature2 = np.loadtxt(self.feature_path+'feature2.txt',dtype=str,delimiter='\t')
        feature3 = np.loadtxt(self.feature_path+'feature3.txt',dtype=str,delimiter='\t')
        data =np.loadtxt(self.pic_info_path,dtype=str,skiprows=1)
        index,category,pic =data[:,0],data[:,1],data[:,2]
        selection_data = open(self.feature_path+'selection_index.txt','w+')
        exclude_data = open(self.feature_path+'exclude_index.txt','w+')
        selection_index = np.random.randint(0, 1000, 200, dtype=int)
        for i in range(1000):
            if( i in selection_index):
                exclude_data.writelines(str(category[i]))
                exclude_data.writelines('\t')
                for j in range(len(feature1[i,:])):
                    exclude_data.writelines(str(feature1[i,j]))    
                    exclude_data.writelines('\t')
                exclude_data.writelines('\n')
            else:    
                selection_data.writelines(str(category[i]))
                selection_data.writelines('\t')
                for j in range(len(feature1[i,:])):
                    selection_data.writelines(str(feature1[i,j]))    
                    selection_data.writelines('\t')
                selection_data.writelines('\n')
        selection_data.close()
        exclude_data.close()
        
    #train
    def train_data(self):
        train = np.loadtxt(self.feature_path+'selection_index.txt',dtype=str,delimiter='\t')
        train_label,train_data = train[:,0].tolist(),train[:,1].tolist()
        predict = np.loadtxt(self.feature_path+'exclude_index.txt',dtype=str,delimiter='\t')
        predict_label,predict_data =predict[:,0].tolist(),predict[:,1].tolist()
        model = libsvm.svm_train(train_label, train_data,'-c 10 -t 0 -p 10')
        p_label, p_acc, p_val = svm_predict(predict_label, predict_data, model)
        

        
if __name__=='__main__':
    pic_info_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Char_Index.txt'
    pic_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Char_Image\\'
    grey_pic_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Gray_Image\\'
    pic_wrong_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\Wrong_index.txt'
    feature_path = 'F:\workspace\deeplearning\Vehicle-License-Plate-Recognition\Vehicle-License-Plate-Recognition\\'

    T = Train(feature_path,pic_info_path)
    #T.write_data()
    T.train_data()
        

        
        
        
            

            
    
