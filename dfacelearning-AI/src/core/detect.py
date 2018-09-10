# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:51:22 2018

@author: Elliott
"""
import torch  as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.core.models import RNet,PNet,ONet

def create_mtcnn_net(p_model_path=None,r_model_path=None,o_model_path=None,use_cuda=True):
    pnet,rnet,onet=None,None,None
    if p_model_path is not None:
        pnet =Pnet(use_cuda=True)
        pnet.load_state_dict(torch.load(p_model_path)) #加载模型
        if(use_cuda):
            pnet.cuda() #使用cuda训练
        pnet.eval()

    if r_model_path is not None:
        rnet =RNet(use_cuda=True)
        rnet.load_state_dict(torch.load(r_model_path))
        if(use_cuda):
            rnet.use_cuda()
        rnet.eval()
        
    if o_model_path is not None:
        onet =ONet(use_cuda=True)
        onet.load_state_dict(torch.load(o_model_path))
        if(use_cuda):
            onet.use_cuda()
        onet.eval()
    
    return pnet,rnet,onet


        
