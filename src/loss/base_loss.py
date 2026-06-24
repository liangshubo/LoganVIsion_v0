import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


                
class BaseLoss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(BaseLoss, self).__init__()
        # Input  Loss function by option 
        self.loss_list = None
        # the lossset_name will control the save loss.pt and load and the pdf 
        self.lossset_name = self.set_loss_savename()
        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.args = args
        self.ckp = ckp
        self.log = torch.Tensor()
        
    @abstractmethod
    def set_loss_savename(self):
        pass

    def load_loss(self):
        if self.loss_list:
            self.loss_module = nn.ModuleList()
            for loss in self.loss_list.split('+'):
                weight, loss_type = loss.split('*')
                loss_function = self.load_baseloss(loss_type)
                self.loss.append({
                    'type': loss_type,
                    'weight': float(weight),
                    'function': loss_function}
                )
                
            if len(self.loss) > 1:
                self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
                
            for l in self.loss:
                if l['function'] is not None:
                    print('\033[1;34m[ =======> Preparing {:} function: {:.3f} * {:} <======= ]\033[0m'.format(self.lossset_name,l['weight'], l['type']))
                    self.loss_module.append(l['function'])
                    
            self.loss_todevice() 
            if self.args.resload_path != '': self.load(self.ckp.dir, cpu=self.args.cpu)
    
    def forward(self,output,label):
        losses = []
        for i, l in enumerate(self.loss):
           
            if l['function'] is not None:
                loss = l['function'](output, label)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        return loss_sum
    
    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()
                
    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))
    
    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)
        
    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)
                
    
    def load_baseloss(self,loss_type):
        if loss_type == 'MSE':
            loss_function = nn.MSELoss()
        elif loss_type == 'L1':
            loss_function = nn.L1Loss()
        elif loss_type == 'CE':
            loss_function = nn.BCELoss()
        else:
            #print("will load ",loss_type)
            loss_function = self.load_extraloss(loss_type)
        return loss_function
    
    @abstractmethod
    def load_extraloss(self,loss_type):
        """
        if you want to used new loss function ,you can rewrite the load_extraloss 
        if loss_type.find('Percept') >= 0:
            module = import_module('loss.perceptualloss')
            loss_function = getattr(module, 'PerceptualLoss')()
        """
        pass
    
    def loss_todevice(self):
        if self.loss_list is not None:
            device = torch.device('cpu' if self.args.cpu else 'cuda')
            self.loss_module.to(device)
            
            if not self.args.cpu and self.args.n_GPUs > 1:
                self.loss_module = nn.DataParallel(
                    self.loss_module, range(self.args.n_GPUs)
                )
            
    def plot_loss(self, apath, epoch):
        # each loss in self.loss will be draw 
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, '{}_{}.png'.format(self.lossset_name,l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module


    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, '{:}.pt'.format(self.lossset_name)))
        torch.save(self.log, os.path.join(apath, '{:}_log.pt'.format(self.lossset_name)))
    
    def load(self,apath , cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        self.load_state_dict(torch.load(
            os.path.join(apath, '{}.pt'.format(self.lossset_name)),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, '{}_log.pt'.format(self.lossset_name)))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()
                
                
