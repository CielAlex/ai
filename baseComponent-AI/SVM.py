# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 22:22:23 2018

@author: Elliott
"""
from libsvm.python.svm import *
from libsvm.python.svmutil import *
import os
def load_data(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [y, x]

	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	prob_x = []
	for line in open(data_file_name,encoding='utf-8'):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		xi = {}
		for e in features.split():
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]
	return (prob_y, prob_x)
if __name__ =='__main__':
    path1=os.path.abspath('.') 
    train_label,train_data =svm_read_problem('train.txt')
    predict_label,predict_data =svm_read_problem('test.txt')
    model =svm_train(train_label,train_data)
    p_label,p_data,p_accuracy =svm_predict(predict_label,predict_data,model)
   # F:\workspace\deeplearning\baseComponent\SVM\train.txt
   git
 
    
    
