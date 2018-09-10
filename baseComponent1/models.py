# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 22:46:50 2018

@author: Elliott
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#basic net example
#nn.Module
#是所有神经网络模块的基类，自定义的网络模块必须继承此模块
#必须重写forward方法，也即前传模块
class Model(nn.Module):
    def __init__():
        super(Model,self).__init__()
        self.conv1 =nn.Conv2d(1,20,kernel_size=5)
        self.conv2 =nn.Conv2d(20,20,kernel_size=5)
    def forward(self,x):
        x =F.relu(self.conv1(x))
        return F.relu(self.conv2(x))    
                

class RNet(nn.Module):
    #定义RNet，卷积池化-卷积池化-卷积输出
    def __init__(self,is_train=False,use_cuda=True):
        super(RNet,self).__init__()
        self.is_train = is_train
        self.use_cude =use_cuda
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,28,kernel_size=3,stride=1),#输入数据通道数、输出数据的通道数、卷积核大小、步长
            nn.PReLU(),#激励函数relu
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(28,48,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(48,64,kernel_size=2,stride=1),
            nn.PReLU()
            )
        self.conv4 =nn.Linear(64*2*2,128)#输入是64*2*2，输出是128线性函数 y=w*x
        self.prelu4 = nn.PReLU()
        self.conv5_1 =nn.Linear(128,1)
        self.conv5_2 =nn.Linear(128,4)
        self.conv5_3 =nn.Linear(128,10)
        self.apply(weights_init)
        
    def forward(self,x):
        x =self.pre_layer(x)
        x =x.view(x.size(0),-1)
        x =self.conv4(x)
        x =self.prelu4(x)
        #detection
        det =torch.sigmod(self.conv5_1(x))
        box =self.conv5_2(x)
        if self.is_train is True:
            return det,box
        return det,box
    
        
class RNet(nn.Module):
    def __init__():
        super(RNet,self).__init__()
        self.is_train = is_train
        self.use_cuda =use_cuda
        self.pre_layer =nn.Sequential(
          nn.Conv2d(3,28,kernel_size=3,stride=1),
          nn.PReLU(),
          nn.MaxPool2d(kernel_size=3,stride=2),
          nn.Conv2d(28,48,kernel_size=1),
          nn.PReLU(),
          nn.MaxPool2d(kernel_size=3,stride=2),
          nn.Conv2d(48,64,kernel_size=1),
          nn.PReLU()
        )
        self.conv4 =nn.Linear(64*2*2,128)
        self.prelu4 =nn.PReLU
        self.conv5_1 =nn.Linear(128,1)
        self.conv5_2 =nn.Linear(128,4)
        self.conv5_3 =nn.Linear(128,10)
        self.apply(weights_init)
    def forward(self,x):
        x =self.pre_layer(x)
        x= x.view(x.size(0)-1)
        x=self.conv4(x)
        x=self.prelu4(x)
        det =torch.sigmod(self.conv5_1(x))
        box =self.conv5_2(x)
        if self.is_train is True:
            return det,box
        return det,box
    
        
        
        
        
        
        
        
        
        
