import os
import time
from decimal import Decimal

from .utility import utility

from tqdm import tqdm
from .evaluation_utils import *

from .utility.make_optimizer import *
from .base_trainer import BaseTrainer

from abc import  abstractmethod

class Trainer_gan(BaseTrainer):
    def __init__(self,args,loader,model_list,loss_list,ckp):
        super(Trainer_gan,self).__init__(args,loader,model_list,loss_list,ckp)
        # Gan have two model two loss and two optimizer 
        self.model, self.model2 = self.get_model(model_list)
        self.loss, self.loss2 = self.get_loss(loss_list)
        self.optimizer,self.optimizer2 = self.get_optimizer(args)
        self.load_optimizer(ckp)

    def get_model(self,model_list):
        model = model_list[0]
        model2 = model_list[1]
        return model, model2
    
    def get_loss(self,loss_list):
        loss = loss_list[0]
        loss2 = loss_list[1]
        return loss ,loss2

    def get_optimizer(self,args):

        optimizer = make_optimizer(args, self.model)
        optimizer2 = make_optimizer2(args,self.model2)
        return optimizer,optimizer2

    def load_optimizer(self,ckp):
        if self.args.resload_path != '':
            print("\033[1;32m[ =======> Optimizer1&Optimizer2 Will load The Save Param   <====== ] \033[0m")
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
            self.optimizer2.load(ckp.dir, epoch=len(ckp.log))

# 

    def loss_step(self):
        self.loss.step()   
    def loss2_step(self):
        self.loss2.step()   
    
    def get_epoch(self):
        epoch = self.optimizer.get_last_epoch() + 1
        return epoch
    
    def get_lr(self):
        lr = self.optimizer.get_lr()
        return lr
        
    def loss_start_log(self):
        self.loss.start_log()

    def loss2_start_log(self):
        self.loss2.start_log()


    @abstractmethod
    def cala_loss(self):
        """
        you should write it for one spectial model the loss function cala,
        this should reach it by pass for skip NotimplementedError
        Args:
            output: the model output
            mask: the label that
        Returns:
        """
        raise NotImplementedError

    @abstractmethod
    def output_process(self,output):
        """
        Args:
            output:
        Returns:
        """
        raise NotImplementedError

    @abstractmethod
    def model_forward(self,image):
        """
        Args:
            image:
        Returns:
        """
        raise NotImplementedError


    def optimizer_zero_grad(self):
        self.optimizer.zero_grad() 
        
    def optimizer2_zero_grad(self):
        self.optimizer2.zero_grad() 
    
    def optimizer_step(self):
        self.optimizer.step()


    def optimizer2_step(self):
        self.optimizer2.step()

    @abstractmethod
    def model_backward(self,image, ground_truth):
        """
        Args:you should write for each model
            image:
            ground_truth:

        Returns:
        """
        raise NotImplementedError
        
    def train_loss_logend(self):
        self.loss.end_log(len(self.loader_train))
        
        self.error_last = self.loss.log[-1, -1]
    
    def train_loss2_logend(self):
        self.loss2.end_log(len(self.loader_train))
        self.error_last2 = self.loss2.log[-1, -1]



    def optimizer_schedule(self):
        self.optimizer.schedule()

    def optimizer2_schedule(self):
        self.optimizer2.schedule()

    @abstractmethod
    def output_quality_test(self):
        #self.ckp.log[-1, idx_data] += utility.calc_skit_psnr(output,ground_truth,rgb_range=self.args.rgb_range)
        raise NotImplementedError

    @abstractmethod
    def output_quality_eval(self):
        #self.ckp.log[-1, idx_data] += utility.calc_skit_psnr(output,ground_truth,rgb_range=self.args.rgb_range)
        #mse = utility.calc_skit_mse(output,ground_truth)
        #ssim = utility.calc_skit_ssim(output,ground_truth,rgb_range=self.args.rgb_range)
        raise NotImplementedError

    def trainepoch_ckp_loss_logstart(self):
        """
        Returns:
        epoch = self.get_epoch()
        lr = self.get_lr()
        self.ckp.write_log('[Epoch {}/{}]\tLearning rate: {:.2e}'.format(epoch, self.epoch, Decimal(lr)))
        self.loss_start_log()
            Returns:
            self.loss.start_log()
        """
        epoch = self.get_epoch()
        lr = self.get_lr()
        self.ckp.write_log('[Epoch {}/{}]\tLearning rate: {:.2e}'.format(epoch, self.epoch, Decimal(lr)))
        self.loss_start_log()
        self.loss2_start_log()

    def testepoch_ckp_logstart(self):
        epoch = self.get_epoch() - 1
        self.test_ckp_logstart(epoch)

    def evlaepoch_ckp_logstart(self):
        epoch = self.get_epoch() - 1
        self.evla_ckp_logstart(epoch)

    def test_epoch_save(self,best):
        epoch = self.get_epoch() - 1
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0,] + 1 == epoch))

    def model_train_control(self):
        self.model.train()
        self.model2.train()
    def model_eval_control(self):
        self.model.eval()
        self.model2.eval()


    def train(self):
        self.loss_step()
        self.loss2_step()
        self.trainepoch_ckp_loss_logstart()
        self.model_train_control()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (image, ground_truth) in enumerate(self.loader_train):
            # - - - - - - - - - - - data_process - - - - - - - - - -
            image, ground_truth = self.prepare(image, ground_truth)
            image, ground_truth = self.down_sample(image, ground_truth)
            # - - - - - - - - - - - start time - - - - - - - - - - -
            timer_data.hold()
            timer_model.tic()
            # - - - - - - - - - - -model forward and loss_optimizer backward
            self.model_backward(image)
            timer_model.hold()
            self.train_ckp_logwrite()
            timer_data.tic()
        self.train_loss_logend()
        self.optimizer_schedule()

    def test(self):
        # 测试阶段只影响的是模型的前向传播过程 ， 以及随输出结果的处理过程，然后就是对评价指标的测量过程
        # 以及要更改对应日志文件输出
        torch.set_grad_enabled(False)
        self.testepoch_ckp_logstart()
        self.model_eval_control()
        timer_test = utility.timer()
        test_dataset_num, total_forward_time = 0, 0
        for idx_data, d in enumerate(self.loader_test):
            test_dataset_num += len(d.dataset)
            single_forward_time = 0
            for image, ground_truth, nameext in tqdm(d, ncols=80):
                start = time.time()
                image, ground_truth = self.prepare(image, ground_truth)

                output = self.model_forward(image)
                output = self.output_process(output)

                torch.cuda.synchronize()
                end = time.time()
                single_forward_time += end - start

                self.output_quality_test()

            total_forward_time += single_forward_time
            self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.test_ckp_logwrite()
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_forward_time,test_dataset_num / total_forward_time))
        self.ckp.write_log('Saving...')
        self.test_epoch_save()
        self.test_ckp_logend(timer_test.toc())
        torch.set_grad_enabled(True)


    def eval(self):
        torch.set_grad_enabled(False)
        self.evlaepoch_ckp_logstart()
        self.model_eval_control()
        timer_test = utility.timer()
        test_dataset_num ,total_evaluation_time = 0,0
        for idx_data, d in enumerate(self.loader_test):
            save_path = self.makdir_path(idx_data)
            test_dataset_num += len(d.dataset)
            evaluation_time = 0 # single dataset evaluation time
            AVGIQA1,AVGIQA2 = 0,0
            count =0
            dataset_sum_iqa1,dataset_sum_iqa2 = 0,0
            for image, ground_true, nameext in tqdm(d, ncols=80):
                image, ground_true = self.prepare(image, ground_true)
                image_d ,ground_true = self.down_sample(image,ground_true)
                t_start = time.time()
                output = self.model_forward(image_d)
                output = self.output_process(output)

                torch.cuda.synchronize()
                t_end = time.time()
                evaluation_time += t_end - t_start

                iqa1,iqa2 = self.output_quality_eval()

                dataset_sum_iqa1 += iqa1
                dataset_sum_iqa2 += iqa2
                count +=1 
                
                result_array = tensor2array(output,self.args.rgb_range)
                self.save_png_output(save_path,nameext[0],result_array)
            
            total_evaluation_time += evaluation_time
            self.ckp.log[-1, idx_data] /= len(d)
            AVGIQA1 = dataset_sum_iqa1/count
            AVGIQA2 = dataset_sum_iqa2/count
            self.evla_ckp_logwrite()
            
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_evaluation_time,test_dataset_num / total_evaluation_time))
        self.ckp.write_log('Saving...')
        self.test_ckp_logend(timer_test.toc())
        torch.set_grad_enabled(True)

    @abstractmethod
    def train_ckp_logwrite(self):
        """
        train_loss_log each step will print
        Returns:
        """
        raise NotImplementedError

    @abstractmethod
    def test_ckp_logwrite(self):
        raise NotImplementedError

    @abstractmethod
    def evla_ckp_logwrite(self):
        raise NotImplementedError

    def terminate(self):
        if self.args.test_only:
            self.eval()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs


'''
if __name__ == '__main__':
    import torch
    import ckpoint
    import data
    import model
    import loss
    from option import args
    from trainer_denoise import Trainer
    import warnings
    warnings.filterwarnings("ignore")
    torch.manual_seed(args.seed)
    ckpoint = ckpoint.ckpoint(args)
    loader = data.Data(args)  # return dataloader which dataset will set by the args
    _model = model.Model(args, ckpoint)  # return model by args
    print('Total params: %.2fM' % (sum(p.numel() for p in
                                       _model.parameters()) / 1000000.0))  # for parameter  in model will cala the parameter number and cala the sum of parameter
    _loss = loss.Loss(args,
                      ckpoint) if not args.test_only else None  # if test_only is true the not true will be false and the loss will become None

    t = Trainer(args, loader, _model, _loss, ckpoint)
    print(len(t.loader_train.dataset))
    
    
    
    
    def train(self):
        self.loss_step()
        epoch = self.get_epoch()
        lr = self.get_lr()
        self.ckp.write_log('[Epoch {}/{}]\tLearning rate: {:.2e}'.format(epoch, self.epoch,Decimal(lr)))
        self.loss_start_log()
        self.model_train_control()
        timer_data, timer_model = ckpoint.timer(), ckpoint.timer()
        for batch, (image, ground_truth) in enumerate(self.loader_train):
            # - - - - - - - - - - - data_process - - - - - - - - - -
            image, ground_truth = self.prepare(image, ground_truth)
            image, ground_truth = self.down_sample(image,ground_truth)
            # - - - - - - - - - - - start time - - - - - - - - - - -
            timer_data.hold()
            timer_model.tic()
            # - - - - - - - - - - -model forward and loss_optimizer backward 
            self.model_backward(image)
            timer_model.hold()
            self.train_loss_log(batch,timer_model,timer_data)
            timer_data.tic()
        self.loss_end_log()
        self.optimizer_schedule()

            
    def test(self):
        #测试阶段只影响的是模型的前向传播过程 ， 以及随输出结果的处理过程，然后就是对评价指标的测量过程 
        #以及要更改对应日志文件输出 
        torch.set_grad_enabled(False)
        epoch = self.get_epoch()-1
        self.test_ckp_log(epoch)
        self.model_eval_control()
        timer_test = ckpoint.timer()
        test_dataset_num ,total_forward_time = 0,0
        for idx_data, d in enumerate(self.loader_test):
            test_dataset_num += len(d.dataset)
            single_forward_time = 0
            for image, ground_truth, nameext in tqdm(d, ncols=80):
                start = time.time()
                image, ground_truth = self.prepare(image,ground_truth)
                output = self.model_forward(image)
                output = self.output_process(output)
                torch.cuda.synchronize()
                end = time.time()
                single_forward_time += end-start
                self.output_quality_assis(idx_data,output,ground_truth)
            total_forward_time += single_forward_time
            self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.test_loss_log(d,idx_data,best)
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format( total_forward_time,test_dataset_num/ total_forward_time))
        self.ckp.write_log('Saving...')
        
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0,] + 1 == epoch))
        self.test_end_log(timer_test.toc())
        torch.set_grad_enabled(True)
        
'''