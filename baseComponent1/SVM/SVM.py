# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:09:54 2018

@author: Elliott
"""
 
from libsvm.python.svm import *
from libsvm.python.svmutil import *

if __name__ =='__main__':
    path1=os.path.abspath('.') 
    train_label,train_data =svm_read_problem('train.txt')
    predict_label,predict_data =svm_read_problem('test.txt')
    model =svm_train(train_label,train_data)
    p_label,p_accuracy,p_data =svm_predict(predict_label,predict_data,model)
    
    
    
