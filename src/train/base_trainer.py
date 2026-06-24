import os
from decimal import Decimal
import cv2
import torch
from tqdm import tqdm
from .utility import utility
import torch.nn.functional as F

import time
from abc import abstractmethod


# mutil model and loss ,so the model and loss will define name by the number 
class BaseTrainer():
    def __init__(self,args,loader,model_list,loss_list,ckp):
        self.args = args
        self.epoch = args.epochs
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.error_last = 1e8
        self.down_scale = args.down_scale
        
        '''
        self.get_model()
        self.get_loss()
        self.get_optimizer()
        self.load_optimizer()
        '''
    
    @abstractmethod
    def get_model(self):
        """
        one model :self.model = model_list[0]
        two model :self.model1 = model_list[0]
                   self.model2 = model_list[1]
        """
        raise NotImplementedError
    @abstractmethod
    def get_loss(self):
        """
        one loss :self.loss1 = loss_list[0]
        two loss :self.loss1 = loss_list[0]
                  self.loss2 = loss_list[1]
        """
        raise NotImplementedError
    @abstractmethod
    def get_optimizer(self):
        """
        one optimizer  :ckpoint.make_optimizer(args, self.model)
        two optimizer  :ckpoint.make_optimizer(args, self.model)
        
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_optimizer(self):
        """
        one optimizer  :ckpoint.make_optimizer(args, self.model)
        two optimizer  :ckpoint.make_optimizer(args, self.model)
        if self.args.resload_path != '':
            print("\033[1;32m[ =======> Optimizer Will load The Save Param   <====== ] \033[0m")
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        """
        raise NotImplementedError
    
    @abstractmethod
    def optimizer_schedule(self):
        """
            self.optimizerg.schedule()
            self.optimizerd.schedule()
        """
        raise NotImplementedError
    
    @abstractmethod
    def cala_loss(self):
        raise NotImplementedError
    
    @abstractmethod
    def output_process(self,output):
        raise NotImplementedError

    @abstractmethod
    def model_forward(self,image):
        raise NotImplementedError
        
    
    @abstractmethod
    def model_backward(self,image, ground_truth):
        '''
        self.optimizer.zero_grad() 
        output = self.model_forward(image)
        loss = self.cala(output)
        loss.backward()
        
        if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(),self.args.gclip)
        self.optimizer.step()
        '''
        raise NotImplementedError
    
    @abstractmethod
    def output_quality_test(self):
        '''
        self.ckp.log[-1, idx_data] += ckpoint.calc_skit_psnr(output,ground_truth,rgb_range=self.args.rgb_range)
        '''
        raise NotImplementedError
    
    def output_quality_eval(self):
        '''
        self.ckp.log[-1, idx_data] += ckpoint.calc_skit_psnr(output,ground_truth,rgb_range=self.args.rgb_range)
        '''
        raise NotImplementedError
    
    def train(self):
        self.loss_step()
        epoch = self.get_epoch()
        lr = self.get_lr()
        self.ckp.write_log('[Epoch {}/{}]\tLearning rate: {:.2e}'.format(epoch, self.epoch,Decimal(lr)))
        self.loss_start_log()
        self.model_train_control()
        
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (image, ground_truth) in enumerate(self.loader_train):
            # - - - - - - - - - - - data_process - - - - - - - - - -
            image, ground_truth = self.prepare(image, ground_truth)
            image, ground_truth = self.down_sample(image,ground_truth)
            # - - - - - - - - - - - start time - - - - - - - - - - -
            timer_data.hold()
            timer_model.tic()
            self.model_backward(image)
            timer_model.hold()
            self.train_ckp_logwrite()
            timer_data.tic()
        self.train_loss_logend()
        self.optimizer_schedule()
            
            
    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.get_epoch()-1
        self.test_ckp_logstrat(epoch)
        self.model_eval_control()
        timer_test = utility.timer()
        test_dataset_num ,total_forward_time = 0,0
        for idx_data, d in enumerate(self.loader_test):
            test_dataset_num += len(d.dataset)
            single_forward_time = 0
            for image, ground_truth, nameext in tqdm(d, ncols=80):
                start = time.time()
                image, ground_truth = self.prepare(image,ground_truth)
                
                self.model_forward(image)
                self.output_process()
                torch.cuda.synchronize()
                end = time.time()
                single_forward_time += end-start
                self.output_quality_test()
                
            total_forward_time += single_forward_time
            self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.test_ckp_logwrite()
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format( total_forward_time,test_dataset_num/ total_forward_time))
        self.ckp.write_log('Saving...')
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0,] + 1 == epoch))
        self.test_end_log(timer_test.toc())
        torch.set_grad_enabled(True)
        
    def eval(self):
        torch.set_grad_enabled(False)
        epoch = self.get_epoch()-1
        self.evla_ckp_log(epoch)
        self.model_eval_control()
        timer_test = utility.timer()
        test_dataset_num ,total_evaluation_time = 0,0
        for idx_data, d in enumerate(self.loader_test):
            save_path = self.makdir_path(idx_data)
            test_dataset_num += len(d.dataset)
            evaluation_time = 0 # single dataset evaluation time
            
            for image, ground_true, nameext in tqdm(d, ncols=80):
                image,ground_true = self.prepare(image, ground_true)
                image_d ,ground_true = self.down_sample(image,ground_true)
                t_start = time.time()
                
                self.model_forward()
                self.output_process()
                torch.cuda.synchronize()
                t_end = time.time()
                evaluation_time += t_end - t_start
                self.output_quality_eval()   
    
                self.save_png_output()
            
            total_evaluation_time += evaluation_time
            self.ckp.log[-1, idx_data] /= len(d)
            self.evla_ckp_logwrite()
            
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_evaluation_time,
                                                                                                   test_dataset_num / total_evaluation_time))
        self.ckp.write_log('Saving...')
        
        self.test_ckp_logend(timer_test.toc())
        torch.set_grad_enabled(True)
                         
            
    @abstractmethod
    def loss_step(self):
        '''
        self.loss.step()
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_epoch(self):
        '''
        epoch = self.optimizer.get_last_epoch() + 1
        
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_lr(self):
        '''
        lr = self.optimizer.get_lr()
        '''
        raise NotImplementedError
    
    
    
    @abstractmethod
    def loss_start_log(self):
        '''
        self.loss.start_log()    
        '''
        raise NotImplementedError
    
    
    @abstractmethod
    def train_loss_logend(self):
        '''
        self.loss.end_log(len(self.loader_train))
        self.loss2.end_log(len(self.loader_train))
        # ADD TWO LOSS
        self.error_last = self.loss.log[-1, -1]
        self.error_last = self.loss2.log[-1, -1]    
        '''
        raise NotImplementedError
    
    
    
    @abstractmethod
    def model_train_control(self):
        '''
        self.model.train()
        '''
        raise NotImplementedError
    
    @abstractmethod
    def model_eval_control(self):
        '''
        self.model.eval()
        '''
        raise NotImplementedError
    
    
    
    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            return tensor.to(device)
        return [_prepare(a) for a in args]
    
    
    def down_sample(self,*args):
        '''
        if args.down_scale != 1 the inout will be down sample
        else return raw image and mask
        '''
        if self.down_scale != 1:
            b, c, h, w = args[0].shape
            #image = F.interpolate(image, (h // self.down_scale, w // self.down_scale))
            #mask = F.interpolate(mask, (h // self.down_scale, w // self.down_scale))
            args =[F.interpolate(a, (h // self.down_scale, w // self.down_scale)) for a in args]
            
        return args
        
        
    @abstractmethod
    def train_ckp_logwrite(self):
        '''
        if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    '[{}/{}]\t{}  {} {:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    # ADD TWO LOSS 
                    self.loss.display_loss(batch),
                    self.loss2.display_loss(batch),
                    # ADD TWO LOSS 
                    timer_model.release(),
                    timer_data.release()))
        '''
        raise NotImplementedError
        
    @abstractmethod
    def test_ckp_logwrite(self):
        '''
        self.ckp.write_log(
                '[{}]\t PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    d.dataset.test_dataset_name,
                    self.ckp.log[-1, idx_data],
                    best[0][idx_data],
                    best[1][idx_data] + 1
                )
            )
        '''
        raise NotImplementedError
    @abstractmethod
    def evla_ckp_logwrite(self):
        """
        best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{}]  PSNR: {:.4f} MSE-e3:{:.4f} SSIM:{:.4f}(BestPSNR: {:.4f} @epoch {})'.format(
                    d.dataset.test_dataset_name,
                    self.ckp.log[-1, idx_data],MSE*1000,SSIM,
                    best[0][idx_data],
                    best[1][idx_data] + 1
                )
            )
        """
        raise NotImplementedError
    
    def test_ckp_logstart(self, epoch):
        self.ckp.write_log('\033[1;32m[=========================================]\033[0m')
        self.ckp.write_log('\033[1;34mTest on Epoch {:}'.format(epoch))
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test)))

    def evla_ckp_logstart(self,epoch):
        self.ckp.write_log('\033[1;32m[=========================================]\033[0m')
        self.ckp.write_log('\033[1;34mEvluation on Epoch {:}'.format(epoch))
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test)))



    def makdir_path(self,idx_data):
        save_path = os.path.join(self.ckp.dir, 'results-{}/Output/'.format(self.args.data_test[idx_data]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path
    
    
    def save_png_output(self,save_path,name,result_array):
        png_save_path = os.path.join(save_path,"Png_Output")
        if not os.path.exists(png_save_path):
            os.makedirs(png_save_path)
        if name.find(".png")<0:
            cv2.imwrite(png_save_path +'/'+ name+'.png', result_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if name.find(".png")>0:
            cv2.imwrite(png_save_path +'/'+ name, result_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    def test_ckp_logend(self,time):
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(time), refresh=True
        )
        self.ckp.write_log('\033[1;32m[=========================================]\033[0m')

    @abstractmethod
    def terminate(self):
        raise NotImplementedError
