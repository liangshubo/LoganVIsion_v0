import os
import time
from decimal import Decimal
from .utility import utility
import numpy as np
import torch.nn.utils as utils
from tqdm import tqdm
from .evaluation_utils import *

from .utility.make_optimizer import *
from .base_trainer_1model_2loss import Trainer_1model_2loss
from torch.utils.tensorboard import SummaryWriter


class Trainer_2loss(Trainer_1model_2loss):
    def __init__(self,args,loader,model_list,loss_list,ckp):
        super(Trainer_2loss,self).__init__(args,loader,model_list,loss_list,ckp)
        self.summarywrite()
        
    def summarywrite(self):
        _,name = self.ckp.dir.split("/experiment/")
        summary_save_path = "/ultrasound/tensorboard_log/liangshubo/" # you need set the path 
        write_path = os.path.join(summary_save_path,name)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        self.writer = SummaryWriter(write_path)
        
        
    # tensorboard add
    def tensorboard_add_scalar(self,name,iqa,batch=None,epoch=None):
        
        if isinstance(iqa, (torch.Tensor, np.ndarray)):
            iqa = iqa.item()
        if batch is not None:
            epochs  = self.get_epoch()
            step = batch+len(self.loader_train)*epochs
            self.writer.add_scalar(name,scalar_value = iqa,global_step = step)          
        elif epoch is not None : 
            self.writer.add_scalar(name,scalar_value = iqa,global_step = epoch)  
        
    def cala_loss(self, output, mask):
        loss1 = self.loss(output, mask)
        loss2 = self.loss2(output, mask)
        loss = loss1 + loss2
        return loss,loss1,loss2

    def output_process(self,output):
        return output
        
    
    def model_forward(self,image):
        output = self.model(image)
        return output


    def model_backward(self,batch,image, ground_truth):
        self.optimizer_zero_grad()
        output = self.model_forward(image)
        output = self.output_process(output)
        loss,loss1,loss2 = self.cala_loss(output, ground_truth)
        loss.backward()
        self.tensorboard_add_scalar("Loss1/Train",loss1,batch=batch)
        self.tensorboard_add_scalar("Loss2/Train",loss2,batch=batch)
        
        if self.args.gclip > 0:
            utils.clip_grad_value_(self.model.parameters(),self.args.gclip)
        self.optimizer_step()
        
        self.writer.flush()
        

    def output_quality_test(self,idx_data,output,ground_truth):
        self.ckp.log[-1, idx_data] += utility.calc_skit_psnr(output, ground_truth, rgb_range=self.args.rgb_range)
    def output_quality_eval(self,idx_data,output,ground_truth):
        self.ckp.log[-1, idx_data] += utility.calc_skit_psnr(output, ground_truth, rgb_range=self.args.rgb_range)
        mse = utility.calc_skit_mse(output, ground_truth)
        ssim = utility.calc_skit_ssim(output, ground_truth, rgb_range=self.args.rgb_range)
        return mse,ssim



    def train(self):
        self.loss_step()
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
            self.model_backward(batch,image,ground_truth)
            timer_model.hold()
            self.train_ckp_logwrite(batch,timer_model,timer_data)
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

                self.output_quality_test(idx_data,output,ground_truth)

            total_forward_time += single_forward_time
            self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.test_ckp_logwrite(d,idx_data,best)
            
            self.tensorboard_add_scalar("PSNR/Test", self.ckp.log[-1, idx_data],epoch=self.get_epoch())
            
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_forward_time,test_dataset_num / total_forward_time))
        self.ckp.write_log('Saving...')
        self.test_epoch_save(best)
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

                iqa1,iqa2 = self.output_quality_eval(idx_data,output,ground_true)

                dataset_sum_iqa1 += iqa1
                dataset_sum_iqa2 += iqa2
                count +=1 
                
                result_array = tensor2array(output,self.args.rgb_range)
                self.save_png_output(save_path,nameext[0],result_array)
            
            total_evaluation_time += evaluation_time
            self.ckp.log[-1, idx_data] /= len(d)
            AVGIQA1 = dataset_sum_iqa1/count
            AVGIQA2 = dataset_sum_iqa2/count
            self.evla_ckp_logwrite(d,idx_data,AVGIQA1,AVGIQA2)
            
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_evaluation_time,test_dataset_num / total_evaluation_time))
        self.ckp.write_log('Saving...')
        self.test_ckp_logend(timer_test.toc())
        torch.set_grad_enabled(True)

    
    def train_ckp_logwrite(self,batch,timer_model,timer_data):
        """
        train_loss_log each step will print
        Returns:
        """
        if (batch + 1) % self.args.print_every == 0:
            self.ckp.write_log(
                    '[{}/{}]\t{} {} {:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    self.loss2.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

    def test_ckp_logwrite(self,d,idx_data,best):
        self.ckp.write_log(
            '[{}]\t PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                d.dataset.test_dataset_name,
                self.ckp.log[-1, idx_data],
                best[0][idx_data],
                best[1][idx_data] + 1
            )
        )

    def evla_ckp_logwrite(self,d,idx_data,MSE,SSIM):
        best = self.ckp.log.max(0)
        self.ckp.write_log(
            '[{}]  PSNR: {:.4f} MSE-e3:{:.4f} SSIM:{:.4f}(BestPSNR: {:.4f} @epoch {})'.format(
                d.dataset.test_dataset_name,
                self.ckp.log[-1, idx_data], MSE * 1000, SSIM,
                best[0][idx_data],
                best[1][idx_data] + 1
            )
        )


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