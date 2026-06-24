# -*- coding: UTF-8 -*-
"""
Create on 2023-10-17
@Author: LiangShubo
@email: liangshubo@neusoftmedical.com

"""
import os
from importlib import import_module
import matplotlib
from PIL.Image import module

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_loss import BaseLoss


# if you want to add some noise ,you can inherit the class and add 
# in self.load_extraloss and must change self.loss_list and self.__lossset_name 



class Loss(BaseLoss):
    def __init__(self, args, ckp):
        super(Loss,self).__init__(args, ckp)
        self.loss_list = args.loss
        self.load_loss()
        
    def set_loss_savename(self):
        loss_savename = "loss"
        return loss_savename
        
    def load_extraloss(self, loss_type):
        #print(loss_type)
        if loss_type.find('Percept') >= 0:
            module = import_module('loss.perceptualloss')
            loss_function = getattr(module, 'PerceptualLoss')()
        elif loss_type.find('Generate') >= 0:
            module = import_module('loss.GenerateLoss')
            loss_function = getattr(module, 'GenerateLoss')()
        elif loss_type.find("Dice") >= 0:
            module = import_module('loss.DiceLoss')
            loss_function = getattr(module,"DiceLoss")()

        elif loss_type.find("BCEDice") >= 0:
            module = import_module('loss.BCEDice_loss')
            loss_function = getattr(module, "BinaryDiceLoss")()

        elif loss_type.find("Softmiou") >=0:
           # print("will load softmiou")
            module = import_module('loss.softiouLoss')   # 损失文件
            loss_function = getattr(module,"IoULoss")()   # 文件内的损失类
        elif loss_type.find("Edge_loss")>=0:
            module = import_module('loss.Edge_loss')
            loss_function = getattr(module,"edge_loss")()

        elif loss_type.find("CrossEntropy") >=0 :
            #weights = torch.tensor([0.5, 1.0, 1.0])
            loss_function = nn.CrossEntropyLoss()
        elif loss_type.find("FocalLoss") >=0 :
            module = import_module('loss.FocalLoss')
            loss_function = getattr(module,"FocalLoss")()
        return loss_function
    

class Loss2(BaseLoss):
    def __init__(self, args, ckp):
        super(Loss2,self).__init__(args, ckp)
        self.loss_list = args.loss2
        self.load_loss()
        
    def set_loss_savename(self):
        loss_savename = "loss2"
        return loss_savename

    def load_extraloss(self, loss_type):
        # print(loss_type)
        if loss_type.find('Percept') >= 0:
            module = import_module('loss.perceptualloss')
            loss_function = getattr(module, 'PerceptualLoss')()
        elif loss_type.find('Generate') >= 0:
            module = import_module('loss.GenerateLoss')
            loss_function = getattr(module, 'GenerateLoss')()

        elif loss_type.find("BCEDice") >= 0:

            module = import_module('loss.BCEDice_loss')
            loss_function = getattr(module, "BinaryDiceLoss")()

        elif loss_type.find("Dice") >= 0:
            module = import_module('loss.DiceLoss')
            loss_function = getattr(module, "DiceLoss")()



        elif loss_type.find("Softmiou") >= 0:
            # print("will load softmiou")
            module = import_module('loss.softiouLoss')  # 损失文件
            loss_function = getattr(module, "IoULoss")()  # 文件内的损失类
        elif loss_type.find("Edge_loss") >= 0:
            module = import_module('loss.Edge_loss')
            loss_function = getattr(module, "edge_loss")()

        elif loss_type.find("CrossEntropy") >= 0:
            # weights = torch.tensor([0.5, 1.0, 1.0])
            loss_function = nn.CrossEntropyLoss()
        elif loss_type.find("FocalLoss") >= 0:
            module = import_module('loss.FocalLoss')
            loss_function = getattr(module, "FocalLoss")()
        return loss_function
    
class Loss3(BaseLoss):
    def __init__(self, args, ckp):
        super(Loss3,self).__init__(args, ckp)
        self.loss_list = args.loss3
        self.load_loss()
        
    def set_loss_savename(self):
        loss_savename = "loss3"
        return loss_savename
        
    def load_extraloss(self, loss_type):
        if loss_type.find('Percept') >= 0:
            module = import_module('loss.perceptualloss')
            loss_function = getattr(module, 'PerceptualLoss')()
        elif loss_type.find('Generate') >= 0:
            module = import_module('loss.GenerateLoss')
            loss_function = getattr(module, 'GenerateLoss')()
        return loss_function
    
