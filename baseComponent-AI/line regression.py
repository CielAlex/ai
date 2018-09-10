# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:28:15 2018

@author: Elliott
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

#定义你的网络
class Net(nn.Module):
    def __init__(self,feature,hidden,output):
        super(Net,self).__init__()
        self.hidden_lay =nn.Linear(feature,hidden)
        self.out_lay =nn.Linear(hidden,output)
    def forward(self,x):
        x =F.relu(self.hidden_lay(x))
        x = self.out_lay(x)
        return x

if __name__ =='__main__':
    #参数准备
    x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
    y = x.pow(2)+0.2*torch.rand(x.size())
    X = Variable(x)
    Y = Variable(y)
    net =Net(1,10,1)
    
    #定义损失函数与优化器
    optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
    loss_func = torch.nn.MSELoss()
    
    plt.ion()#开启交互模式
    for i in range(1000):
        prediction = net(x)
        optimizer.zero_grad() #
        loss_func(prediction,Y).backward()
        optimizer.step()#更新参数
        #打印
        if i%5 ==0:
            plt.cla()
            plt.scatter(x.data.numpy(),y.data.numpy())
            plt.plot(x.data.numpy(),prediction.data.numpy())
            plt.pause(0.1)
    plt.ioff()
    plt.show()
    

        
        
        
    
    
    
        