# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:58:17 2018

@author: 烽
"""
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
class Net(nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden =nn.Linear(n_features,n_hidden)
        self.output =nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
    
if __name__=='__main__':
    #data
    x_data =torch.ones(100,2)
    x0 = torch.normal(2*x_data,1)
    x1 = torch.normal(-2*x_data,1)
    y0 = torch.ones(100)
    y1 = torch.zeros(100)
    x  = torch.cat((x0,x1),0).type(torch.FloatTensor)
    y  = torch.cat((y0,y1),0).type(torch.LongTensor)
    x,y = Variable(x),Variable(y)
    net =Net(2,10,2)
    
    #optimizer loss_func
    optimizer =torch.optim.SGD(net.parameters(),lr=0.1)
    loss_func =torch.nn.CrossEntropyLoss()
    
    optimizer =torch.optim.SGD()
    
    #train
    plt.ion()
    for i in range(1000):
        out =net(x)
        loss  =loss_func(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        out =net(x)
        loss=loss_func(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%5==0:
            plt.cla()   
            prediction =torch.max(F.softmax(out),1)[1]
            pre_y =prediction.data.numpy().squeeze()
            target_y =y.data.numpy()
            plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1])
            accuracy=sum(pre_y==target_y)/200
            plt.text(1.5,-4,'Accurancy=%.2f'%accuracy,fontdict={'size':20,'color':'red'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()
        
        
        
        
        
        