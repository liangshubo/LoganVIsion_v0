import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import torch


from .base_checkpoint import Basecheckpoint

class checkpoint_1model_1loss(Basecheckpoint):
    def __init__(self, args) -> None:
        super(checkpoint_1model_1loss,self).__init__(args)
        
    def save(self,trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best) #save the self.dir / model/epoch
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        self.plot_metric(epoch) #draw the epoch psnr
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('train-test-log.pt'))

class checkpoint_1model_2loss(Basecheckpoint):
    def __init__(self, args) -> None:
        super(checkpoint_1model_2loss,self).__init__(args)
        
    def save(self,trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best) #save the self.dir / model/epoch
        
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        
        trainer.loss2.save(self.dir)
        trainer.loss2.plot_loss(self.dir, epoch)
        
        
        self.plot_metric(epoch) #draw the epoch psnr
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('train-test-log.pt'))


class checkpoint_2model_2loss(Basecheckpoint):
    def __init__(self, args) -> None:
        super(checkpoint_2model_2loss,self).__init__(args)
        
    def save(self,trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best) #save the self.dir / model/epoch
        trainer.model2.save(self.get_path('model'), epoch, is_best=is_best) #save the self.dir / model/epoch
        
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        
        trainer.loss2.save(self.dir)
        trainer.loss2.plot_loss(self.dir, epoch)
        
        self.plot_metric(epoch) #draw the epoch psnr
        trainer.optimizer.save(self.dir)
        trainer.optimizer2.save(self.dir)
        torch.save(self.log, self.get_path('train-test-log.pt'))
