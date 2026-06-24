import os
import time
import cv2
import utility.utility as utility
import torch.nn.utils as utils
from tqdm import tqdm
import torch.nn.functional as F
from train.evaluation_utils import *



from utility.make_optimizer import *

from train.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self,args,loader,model_list,loss_list,ckp):
        super(Trainer,self).__init__(args,loader,model_list,loss_list,ckp)
        self.model = self.get_model(model_list)
        self.loss = self.get_loss(loss_list)
        self.optimizer = self.get_optimizer(args)
        self.load_optimizer(ckp)
    
    def get_model(self,model_list):
        model = model_list[0]
        return model

    def get_loss(self,loss_list):
        loss = loss_list[0]
        return loss 

    def get_optimizer(self,args):
        optimizer = make_optimizer(args, self.model)
        return optimizer

    def load_optimizer(self,ckp):
        if self.args.resload_path != '':
            print("\033[1;32m[ =======> Optimizer Will load The Save Param   <====== ] \033[0m")
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        
    def loss_step(self):
        self.loss.step()   
    
    def get_epoch(self):
        epoch = self.optimizer.get_last_epoch() + 1
        return epoch
    
    def get_lr(self):
        lr = self.optimizer.get_lr()
        return lr
        
    def loss_start_log(self):
        self.loss.start_log()    

    def cala_loss(self,output,mask):
        loss = self.loss(output, mask)
        return loss
        
    def output_process(self,output):
        return output
        
    def model_forward(self,image):
        output = self.model(image)
        return output
        
    def optimizer_zero_grad(self):
        self.optimizer.zero_grad() 
        
    def optimizer_step(self):
        self.optimizer.step()
        
    def model_backward(self,image, ground_truth):
        self.optimizer_zero_grad()
        output = self.model_forward(image)
        output = self.output_process(output)
        loss = self.cala_loss(output,ground_truth)
        loss.backward()
        
        if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(),self.args.gclip)
        self.optimizer_step()
        
    def loss_end_log(self):
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
    
    def optimizer_schedule(self):
        self.optimizer.schedule()

    def test_ckp_log(self,epoch):
        self.ckp.write_log('\033[1;32m[=========================================]\033[0m')
        self.ckp.write_log('\033[1;34mEvaluation on Epoch {:}'.format(epoch))
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test)))
        
    def output_quality_test(self,idx_data,output,ground_truth):
        self.ckp.log[-1, idx_data] += utility.calc_skit_psnr(output, ground_truth, rgb_range=self.args.rgb_range)

    def output_quality_eval(self,idx_data,output,ground_truth):
        self.ckp.log[-1, idx_data] += utility.calc_skit_psnr(output, ground_truth, rgb_range=self.args.rgb_range)
        mse = utility.calc_skit_mse(output, ground_truth)
        ssim = utility.calc_skit_ssim(output, ground_truth, rgb_range=self.args.rgb_range)
        return mse,ssim 
        
        
    def eval(self):
        torch.set_grad_enabled(False)
        epoch = self.get_epoch()-1
        self.test_ckp_log(epoch)
        self.model_eval_control()
        timer_test = utility.timer()
        test_dataset_num ,total_evaluation_time = 0,0
        for idx_data, d in enumerate(self.loader_test):
            save_path = self.makdir_path(idx_data)
            test_dataset_num += len(d.dataset)
            evaluation_time = 0 # single dataset evaluation time
            MSE,SSIM = 0,0
            count =0
            dataset_mse,dataset_ssim = 0,0
            for image, ground_true, nameext in tqdm(d, ncols=80):
                image,ground_true = self.prepare(image, ground_true)
                image_d ,ground_true = self.down_sample(image,ground_true)
                t_start = time.time()
                output = self.model_forward(image_d)
                output = self.output_process(output)
                torch.cuda.synchronize()
                t_end = time.time()
                evaluation_time += t_end - t_start
                mse,ssim = self.output_quality_eval(idx_data,output,ground_true)   
                dataset_mse += mse
                dataset_ssim += ssim 
                count +=1 
                
                result_array = tensor2array(output,self.args.rgb_range)
                self.save_png_output(save_path,nameext[0],result_array)
            
            total_evaluation_time += evaluation_time
            self.ckp.log[-1, idx_data] /= len(d)
            MSE = dataset_mse/count
            SSIM = dataset_ssim/count
            self.eval_loss_log(self,d,idx_data,MSE,SSIM)
            
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_evaluation_time,test_dataset_num / total_evaluation_time))
        self.ckp.write_log('Saving...')
        self.test_end_log(timer_test.toc())
        torch.set_grad_enabled(True)
    
    
    
    def train_loss_log(self,batch,timer_model,timer_data):
        if (batch + 1) % self.args.print_every == 0:
            self.ckp.write_log(
                    '[{}/{}]\t{} {:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    # ADD TWO LOSS 
                    timer_model.release(),
                    timer_data.release()))
    
    def test_loss_log(self,d,idx_data,best):
        self.ckp.write_log(
                '[{}]\t PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    d.dataset.test_dataset_name,
                    self.ckp.log[-1, idx_data],
                    best[0][idx_data],
                    best[1][idx_data] + 1
                )
            )
        
    def eval_loss_log(self,d,idx_data,MSE,SSIM):
        best = self.ckp.log.max(0)
        self.ckp.write_log(
                '[{}]  PSNR: {:.4f} MSE-e3:{:.4f} SSIM:{:.4f}(BestPSNR: {:.4f} @epoch {})'.format(
                    d.dataset.test_dataset_name,
                    self.ckp.log[-1, idx_data],MSE*1000,SSIM,
                    best[0][idx_data],
                    best[1][idx_data] + 1
                )
            )
        
    def model_train_control(self):
        self.model.train()
        
    def model_eval_control(self):
        self.model.eval()

    
    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            return tensor.to(device)
        return [_prepare(a) for a in args]
    
    def down_sample(self,image,mask):
        '''
        if args.down_scale != 1 the inout will be down sample
        else return raw image and mask
        '''
        if self.down_scale != 1:
            b, c, h, w = image.shape
            image = F.interpolate(image, (h // self.down_scale, w // self.down_scale))
            mask = F.interpolate(mask, (h // self.down_scale, w // self.down_scale))
        return image,mask

    def makdir_path(self,idx_data):
        save_path = os.path.join(self.ckp.dir, 'results-{}/Output/'.format(self.args.data_test[idx_data]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path
    
    def save_png_output(self,save_path,name,result_array):
        png_save_path = os.path.join(save_path,"Png_Output")
        if not os.path.exists(png_save_path):
            os.makedirs(png_save_path)
        cv2.imwrite(png_save_path +'/'+ name+'.png', result_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def test_end_log(self,time):
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(time), refresh=True
        )
        self.ckp.write_log('\033[1;32m[=========================================]\033[0m')
        
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
    
    
    
    
    #def train(self):
        #self.loss_step()
        #epoch = self.get_epoch()
        #lr = self.get_lr()
        #self.ckp.write_log('[Epoch {}/{}]\tLearning rate: {:.2e}'.format(epoch, self.epoch,Decimal(lr)))
        #self.loss_start_log()
        #self.model_train_control()
        #timer_data, timer_model = ckpoint.timer(), ckpoint.timer()
        #for batch, (image, ground_truth) in enumerate(self.loader_train):
            # - - - - - - - - - - - data_process - - - - - - - - - -
            #image, ground_truth = self.prepare(image, ground_truth)
            #image, ground_truth = self.down_sample(image,ground_truth)
            # - - - - - - - - - - - start time - - - - - - - - - - -
            #timer_data.hold()
            #timer_model.tic()
            # - - - - - - - - - - -model forward and loss_optimizer backward 
            #self.model_backward(image)
            #timer_model.hold()
            #self.train_loss_log(batch,timer_model,timer_data)
            #timer_data.tic()
        #self.loss_end_log()
        #self.optimizer_schedule()

            
    #def test(self):
        #测试阶段只影响的是模型的前向传播过程 ， 以及随输出结果的处理过程，然后就是对评价指标的测量过程 
        #以及要更改对应日志文件输出 
        #torch.set_grad_enabled(False)
        #epoch = self.get_epoch()-1
        #self.test_ckp_log(epoch)
        #self.model_eval_control()
        #timer_test = ckpoint.timer()
        #test_dataset_num ,total_forward_time = 0,0
        #for idx_data, d in enumerate(self.loader_test):
            #test_dataset_num += len(d.dataset)
            #single_forward_time = 0
            #for image, ground_truth, nameext in tqdm(d, ncols=80):
                #start = time.time()
                #image, ground_truth = self.prepare(image,ground_truth)
                #output = self.model_forward(image)
                #output = self.output_process(output)
                #torch.cuda.synchronize()
                #end = time.time()
                #single_forward_time += end-start
                #self.output_quality_assis(idx_data,output,ground_truth)
            #total_forward_time += single_forward_time
            #self.ckp.log[-1, idx_data] /= len(d)
            #best = self.ckp.log.max(0)
            #self.test_loss_log(d,idx_data,best)
        #self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format( total_forward_time,test_dataset_num/ total_forward_time))
        #self.ckp.write_log('Saving...')
        
        #if not self.args.test_only:
            #self.ckp.save(self, epoch, is_best=(best[1][0,] + 1 == epoch))
        #self.test_end_log(timer_test.toc())
        #torch.set_grad_enabled(True)
        
'''