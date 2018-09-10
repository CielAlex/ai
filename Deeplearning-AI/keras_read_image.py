# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:30:15 2018
采用的是Olivetti Faces，纽约大学的一个比较小的人脸库，一共是40个人，每个人10
@author: Elliott
"""
from PIL import Image
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import  Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

ndarray = np.array([[1,2],[3,4]])
faces = np.empty((400,2679));
#图片大小
img_rows = 57;
img_cols = 47;
#类别
nb_classes = 40;
#卷积和的数目
nb_filters1, nb_filters2 = 5, 10  
#卷积和的宽度与长度 
nb_conv = 3
#下载图片
def load_image(imagePath):
    #读取图片942 = 20* 47 1140 = 20* 57
    image = Image.open(imagePath);
    #将图片数组转换成ndarray对象
    image_ndarray = np.asarray(image,dtype = 'float64');
    
    #400张图片
    #图片大小是 57*47 =2679
    for row in range(20):
        for column in range(20):
            faces[row+20*column]=np.ndarray.flatten(image_ndarray[row*57:(row+1)*57,column*47:(column+1)*47]);
    lable = np.empty(400)
    for i in range(40):
        lable[i*10:i*10+10] = i;
    lable = lable.astype(np.int);  
        
    #训练样本
    train_data = np.empty((320,2679));
    train_lable = np.empty(320);
    #内测
    valid_data = np.empty((40,2679));
    valid_lable = np.empty(40);
    #公测
    test_data = np.empty((40,2679));
    test_lable = np.empty(40);
    #抽取数据，其中一共40人，每个人10个图片，8个用于训练，一个内测，一个公测                     
    for i in range(40):
        test_data[i,0:2679] = faces[i*10,0:2679];
        test_lable[i] = lable[i*10];
        valid_data[i,0:2679] = faces[(i*10+1),0:2679];
        valid_lable[i] = lable[(i*10+1)];
        train_data[(i*8):(i*8+8),0:2679] = faces[(i*10+2):(i*10+10),0:2679];
        train_lable[(i*8):(i*8+8)] = lable[(i*10+2):(i*10+10)]
    train_data = train_data.astype('float32');
    valid_data = valid_data.astype('float32');
    test_data = test_data.astype('float32');
    reval = [(train_lable, train_data),(valid_lable,valid_data),(test_lable,test_data)];
    return reval;
    
#训练  
def train_model(model,X_train, Y_train, X_val, Y_val):  
    model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,  
          verbose=1, validation_data=(X_val, Y_val))  
    model.save_weights('model_weights.h5', overwrite=True)  
    return model

#公测
def test_model(model,X,Y):  
    model.load_weights('model_weights.h5')  
    score = model.evaluate(X, Y, verbose=0)
    return score  

#设置训练模型
def set_model(lr = 0.005,decay = 1e-6, momentum = 0.9 ):
    model = Sequential();
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(nb_filters1, kernel_size=(nb_conv, nb_conv), input_shape = (1, img_rows, img_cols)))
    else:
        model.add(Conv2D(nb_filters1, kernel_size=(nb_conv, nb_conv), input_shape = (img_rows, img_cols, 1)))
    #添加激励函数
    model.add(Activation('tanh'))
    #池化
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))  
    #再次卷积
    model.add(Conv2D(10, kernel_size=(nb_conv, nb_conv)))  
    #添加激励函数
    model.add(Activation('tanh'))  
    #池化
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))  
    #dropout
    model.add(Dropout(0.25))  
    #flatten
    model.add(Flatten())  
    model.add(Dense(1000)) #Full connection  
    model.add(Activation('tanh'))  
    model.add(Dropout(0.5))  
    model.add(Dense(nb_classes))  
    model.add(Activation('softmax'))  
  
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd) 
        
    return model

if __name__ == '__main__':
    #读取图片
    (train_lable,train_data),(valid_lable,valid_data),(test_lable,test_data)=load_image('olivettifaces.gif');
    #判断这版本的keras接受的图片是通道在前还是在后
    #将数据变成320 57 47 1
    #          40  57 47 1
    #          40  57 47 1 格式
    if K.image_data_format() == 'channels_first':
        train_data = train_data.reshape(train_data.shape[0],1,img_rows,img_cols);
        valid_data = valid_data.reshape(valid_data.shape[0],1,img_rows,img_cols);
        test_data = test_data.reshape(test_data.shape[0],1,img_rows,im_cols);
        input_shape = (1,img_rows,img_cols);
    else:
        train_data = train_data.reshape(train_data.shape[0],img_rows,img_cols,1);
        valid_data = valid_data.reshape(valid_data.shape[0],img_rows,img_cols,1);
        test_data = test_data.reshape(test_data.shape[0],img_rows,img_cols,1);
        input_shape = (img_rows,img_cols,1);
    #将标签进行编码
    #如：1变成10000000 2 0200000
    train_Lable = np_utils.to_categorical(train_lable,nb_classes);
    valid_Lable = np_utils.to_categorical(valid_lable,nb_classes);
    test_Lable = np_utils.to_categorical(test_lable,nb_classes);
    
    model = set_model();
    train_model(model, train_data, train_Lable, valid_data, valid_Lable)   
    score = test_model(model, test_data, test_Lable)  
    
    model.load_weights('model_weights.h5')  
    classes = model.predict_classes(test_data, verbose=0)  
    test_accuracy = np.mean(np.equal(test_Lable, classes))  
    print("accuarcy:", test_accuracy)
    for i in range(0,40):
        if y_test[i] != classes[i]:
            print(y_test[i], '被错误分成', classes[i]);
        














