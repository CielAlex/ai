# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 11:21:30 2018

@author: Elliott
"""
import matplotlib.pyplot as plt
import torch

if __name__=='__main__':
    x = torch.linspace(-1,1,100)  #tensor type -1 1 100个数
    y = x.pow(2) + 0.2*torch.rand(x.size())  #0,1 里面产生x.size()个随机数
    plt.ion() #交互模式
    for i in range(1000):
        plt.cla() #清空画板
        plt.plot(x.data.numpy(),y.data.numpy()) #tensor转换成array格式输出 ,连线
        plt.scatter(x.data.numpy(),y.data.numpy()) #点图
        plt.pause(0.1) #停留0.1s
    plt.ioff() #关闭
    plt.show()
            
    
    
    
        