import os
import time
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
import cv2
from .utility import utility
#import utility.utility as utility
import torch.nn.utils as utils
from tqdm import tqdm
import numpy as np 
from .evaluation_utils import *

from .utility.make_optimizer import *
from .base_trainer_1model_1loss import Trainer_1model_1loss
from torch.utils.tensorboard import SummaryWriter

# 适用于分类问题的训练框架的代码 ，不同的将是评估时候的评价指标与评价方法
class Trainer(Trainer_1model_1loss):
    def __init__(self,args,loader,model_list,loss_list,ckp):
        super(Trainer, self).__init__(args,loader,model_list,loss_list,ckp)
        self.summarywrite()
        # 用于评估时候的相关指标计算标注
        self.class_name = ["S0","S1","S2","S3","S4","S5","S6","S7","S8","S10","S11"]
        self.evaluator_index = utility.ClassificationEvaluator(args.num_class,class_names=self.class_name)
    # tensorboard init 
    def summarywrite(self):
        
        _,name = self.ckp.dir.split("/experiment/")
        summary_save_path = r"/home/ubuntu4090/4T_disk/liangshubo/Sci_Plus/tensor"
        write_path = os.path.join(summary_save_path,name)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        self.writer = SummaryWriter(write_path)
    # tensorboard add
    def tensorboard_add_scalar(self,name,iqa,batch):
        epoch  = self.get_epoch()
        
        if isinstance(iqa, (torch.Tensor, np.ndarray)):
            iqa = iqa.item()
        self.writer.add_scalar(name,scalar_value = iqa,global_step = batch+len(self.loader_train)*epoch)          
        
    def cala_loss(self, output, mask):
        loss = self.loss(output, mask)
        return loss

    def output_process(self,output):
        return output

    def model_forward(self,image):
        output = self.model(image)
        return output

    def model_backward(self,batch,image, ground_truth):
        self.optimizer_zero_grad()
        output = self.model_forward(image)
        output = self.output_process(output)
        loss = self.cala_loss(output, ground_truth)
        loss.backward()
        
        self.tensorboard_add_scalar("loss/train",loss,batch)
        self.writer.flush()
        
        
        if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(),self.args.gclip)
        self.optimizer_step()
        

    def train(self):
        self.loss_step()
        self.trainepoch_ckp_loss_logstart()
        self.model_train_control()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (image, ground_truth) in enumerate(self.loader_train):
            
            #print(image.shape,ground_truth.shape)
            # - - - - - - - - - - - data_process - - - - - - - - - -
            image, ground_truth= self.prepare(image, ground_truth)
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
            true_num = 0
            total_num = 0
            for image, ground_truth, nameext in tqdm(d, ncols=80):
                start = time.time()

                image, ground_truth = self.prepare(image, ground_truth)

                output = self.model_forward(image)
                output = self.output_process(output)
                torch.cuda.synchronize()
                end = time.time()
                single_forward_time += end - start
                true_num,total_num = self.output_quality_test(idx_data,output,ground_truth,true_num,total_num)
            total_forward_time += single_forward_time
            #self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.test_ckp_logwrite(d,idx_data,best)
            
            # tensorboard 
            self.tensorboard_add_scalar("Avg_Accuracy/Test_{}".format(idx_data),self.ckp.log[-1, idx_data],self.get_epoch())
            self.writer.flush()

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
            for image, ground_true, nameext in tqdm(d, ncols=80):
                image, ground_true = self.prepare(image, ground_true)

                t_start = time.time()
                output = self.model_forward(image)
                output = self.output_process(output)
                if not self.args.cpu:
                    torch.cuda.synchronize()
                t_end = time.time()
                evaluation_time += t_end - t_start
                # 对于分类这个要改一下，显示图像，标注预测类和实际类 ，而且不是用的
                input_array = tensor2array(image,self.args.rgb_range)
                self.save_png_output(save_path,nameext[0],input_array,ground_true,output)
                self.output_quality_eval(ground_true, output)

            total_evaluation_time += evaluation_time
            _,evluation_result=self.evaluator_index.evaluate(save_path)
            self.ckp.write_log(evluation_result)
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_evaluation_time,test_dataset_num / total_evaluation_time))
        self.ckp.write_log('Saving...')
        self.test_ckp_logend(timer_test.toc())
        torch.set_grad_enabled(True)

    def prepare(self, image,label):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            return tensor.to(device)

        return _prepare( image) ,_prepare(label)

    def save_png_output(self,save_path,name,result_array,ground_true,output):

        # 确保图像是3通道（如果是灰度图则转换为BGR）
        if len(result_array.shape) == 2:  # 灰度图 [H,W]
            result_array = cv2.cvtColor(result_array, cv2.COLOR_GRAY2BGR)
        elif result_array.shape[2] == 1:  # 单通道 [H,W,1]
            result_array = cv2.cvtColor(result_array, cv2.COLOR_GRAY2BGR)

        true_class = ground_true.cpu().item()
        pred_class = np.argmax(np.array(output.cpu()))
        #print(true_class,pred_class)
        layer = nn.Softmax()
        confidence = layer(output)[:,pred_class]

        true_class_name = self.class_name[true_class]
        pred_class_name = self.class_name[pred_class]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_color = (255, 255, 255)  # 白色文字
        true_color = (0, 255, 0)  # 绿色表示真实类别
        pred_color = (0, 165, 255)  # 橙色表示预测类别

        # 在图像顶部添加文字
        y_offset = 30
        line_height = 30

        # 添加真实类别标注
        true_text = f"Label: {true_class_name}"
        cv2.putText(result_array, true_text, (10, y_offset),
                    font, font_scale, true_color, thickness, cv2.LINE_AA)

        # 添加预测类别标注
        pred_text = f"Pred: {pred_class_name} ({confidence.item():.2f})"
        cv2.putText(result_array, pred_text, (10, y_offset + line_height),
                    font, font_scale, pred_color, thickness, cv2.LINE_AA)

        # 如果预测错误，添加红色警告
        if true_class != pred_class:
            warning_text = "(!) Prediction Incorrect"
            cv2.putText(result_array, warning_text, (10, y_offset + 2 * line_height),
                        font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        png_save_path = os.path.join(save_path,"Png_Output")
        if not os.path.exists(png_save_path):
            os.makedirs(png_save_path)
        if name.find(".png")<0:
            cv2.imwrite(png_save_path +'/'+ name+'.png', result_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if name.find(".png")>0:
            cv2.imwrite(png_save_path +'/'+ name, result_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def output_quality_test(self,idx_data,output,ground_truth,true_num,total_num):
         # 测试数集指标 平均准确度、平均精确度 、平均召回率
        self.ckp.log[-1, idx_data] ,true_num,total_num = utility.calc_accuray(output, ground_truth,true_num,total_num)
        return true_num,total_num
    def output_quality_eval(self,ground_truth,output):
        # 这里只进行更新

        self.evaluator_index.update(ground_truth,output)
        return

    def train_ckp_logwrite(self,batch,timer_model,timer_data):
        '''
        类内隐保存当前损失数值
        Args:
            batch:
            timer_model:
            timer_data:
        Returns:
        '''
        if (batch + 1) % self.args.print_every == 0:
            self.ckp.write_log(
                    '[{}/{}]\t{} {:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

    def test_ckp_logwrite(self,d,idx_data,best):
        self.ckp.write_log(
            '[{}]\t Avg_Accuracy: {:.3f} (Best: {:.3f} @epoch {})'.format(
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
