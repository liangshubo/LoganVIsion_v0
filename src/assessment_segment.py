
import data

import model
import ckpoint as ckp
from tqdm import tqdm
import train
import os
from train.trainer_segment import Trainer
import torch
import loss
from train.evaluation_utils import *

from train.utility import utility
import time 


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2

def tensor2arrayf(image_tensor):
    image_tensor = image_tensor.squeeze(0).squeeze(0).detach()
    image = np.array(image_tensor)
    return image

def log_compress(image_mask_array,c=0.8):
    image = c*np.log(image_mask_array+1)
    return image

def linear_compress(image_mask_array,c=0.5):
    image = c*image_mask_array
    return image


def postimage(image_array,mask_array):
    for i in range(5):
        mask_array = cv2.GaussianBlur(mask_array,(5,5),sigmaX=1,sigmaY=1)
    re_mask_array = 1 - mask_array
    image_inter = image_array * mask_array
    image_inter = linear_compress(image_inter)

    image_exter = image_array * re_mask_array

    image = image_exter + image_inter
    return image


def biliary_mask(mask_tensor,shold=0.1):
    condition = mask_tensor>shold
    mask_tensor[condition]=1
    mask_tensor[~condition]=0
    return mask_tensor

def show_mask_image(image_array,mask_array):
    image = image_array + mask_array*0.5
    return image


def mcc_edge(img):
    """
    Extract max connected component and then extract edge.
    """
    # 代码要求为255 最大数值 而且是array形式  而且必须是单通道的格式
    #print(in_img.dtype)
    #print(in_img.max())
    in_img=img

    if in_img.dtype != 'uint8':
        in_img = in_img * 255
        img = in_img.astype('uint8')

    # Max connected component extraction
    # 调用的cv2的这函数是什么 鬼东西
    # 要求为8单通道的图像
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    # 返回联通区域的数量，连通区域的背景分割标签图，联通区域的外接矩形框，联通区域的质心
    sort_label = np.argsort(-stats[:, 4], )
    idx = labels == sort_label[1]
    # 将最大的联通区域使用True 标注
    max_connect = idx * 255
    max_connect = max_connect.astype('uint8')
    #得到一个矩阵，是最大的联通区域
    # Edge detection
    #edge_img = cv2.Canny(max_connect, 50, 250)

    contours, _ = cv2.findContours(max_connect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 提取最大连通区域的边缘
    filled_result = cv2.fillPoly(max_connect.copy(),   contours, color=255)
    return filled_result



class Assessmen(Trainer):
    def __init__(self,args,loader,model_list,loss_list,ckp):
        super(Assessmen,self).__init__(args,loader,model_list,loss_list,ckp)
    def assement(self):
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
            
            f = open(os.path.join('/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/experiment',
                                  self.args.pre_train, "Evluation","results-"+d.dataset.test_dataset_name,"iqa_log.txt"),'w')
            IQA_0_DIC = {}
            IQA_1_DIC = {}
            IQA_2_DIC = {}
            for image, ground_true, nameext in tqdm(d, ncols=80):
                image, ground_true = self.prepare(image, ground_true)
                #image_d ,ground_true = self.down_sample(image,ground_true)
                t_start = time.time()
                output = self.model_forward(image)
                output = self.output_process(output)
                if not self.args.cpu:
                    torch.cuda.synchronize()
                t_end = time.time()
                evaluation_time += t_end - t_start

                iqa0,iqa1,iqa2 = self.output_quality_eval(idx_data,output,ground_true)
                IQA_0_DIC[nameext[0]] = iqa0
                IQA_1_DIC[nameext[0]] = iqa1
                IQA_2_DIC[nameext[0]] = iqa2
                
                
                dataset_sum_iqa1 += iqa1
                dataset_sum_iqa2 += iqa2
                count +=1 
                
                result_array = tensor2array(output,self.args.rgb_range)
                self.save_png_output(save_path,nameext[0],result_array)
            # ---
            SORT_IQA_0_DIC = dict(sorted(IQA_0_DIC.items(), key=lambda item: item[1], reverse=True))
            for filename,iqa0 in SORT_IQA_0_DIC.items():
                info = "FileName: [{:}]  Dice: [{:.4f}]  MSE:[{:.4f}]  Softmiou:[{:.2f}] ".format(filename,IQA_0_DIC[filename],IQA_1_DIC[filename],IQA_2_DIC[filename])
                f.write(info+"\n")
            f.close()
                
            
            # ---
            
            
            total_evaluation_time += evaluation_time
            self.ckp.log[-1, idx_data] /= len(d)
            AVGIQA1 = dataset_sum_iqa1/count
            AVGIQA2 = dataset_sum_iqa2/count
            self.evla_ckp_logwrite(d,idx_data,AVGIQA1,AVGIQA2)
            
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_evaluation_time,test_dataset_num / total_evaluation_time))
        self.ckp.write_log('Saving...')
        self.test_ckp_logend(timer_test.toc())
        torch.set_grad_enabled(True)


    def post_save_png_output(self,save_path,folder,name,result_array):
        png_save_path = os.path.join(save_path,folder)
        if not os.path.exists(png_save_path):
            os.makedirs(png_save_path)
        #print(png_save_path,name)
        if name.find(".png")<0:
            cv2.imwrite(png_save_path +'/'+ name+'.png', result_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if name.find(".png")>=0:
            cv2.imwrite(png_save_path +'/'+ name, result_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def post_process(self):
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
            
            f = open(os.path.join('/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/experiment',
                                  self.args.pre_train, "Evluation","results-"+d.dataset.test_dataset_name,"iqa_log.txt"),'w')
            IQA_0_DIC = {}
            IQA_1_DIC = {}
            IQA_2_DIC = {}
            for image, ground_true, nameext in tqdm(d, ncols=80):
                image, ground_true = self.prepare(image, ground_true)
                #image_d ,ground_true = self.down_sample(image,ground_true)
                t_start = time.time()
                output = self.model_forward(image)
                output = self.output_process(output)
                if not self.args.cpu:
                    torch.cuda.synchronize()
                t_end = time.time()
                evaluation_time += t_end - t_start

                iqa0,iqa1,iqa2 = self.output_quality_eval(idx_data,output,ground_true)
                IQA_0_DIC[nameext[0]] = iqa0
                IQA_1_DIC[nameext[0]] = iqa1
                IQA_2_DIC[nameext[0]] = iqa2
                
                
                dataset_sum_iqa1 += iqa1
                dataset_sum_iqa2 += iqa2
                count +=1 
                # ----- 
                mask_tensor = biliary_mask(output)
                mask_array = tensor2arrayf(mask_tensor.cpu())
                mask_array  = mcc_edge(mask_array*255)/255
                image_array = tensor2arrayf(image.cpu())
                
                post_image =postimage(image_array,mask_array)
                #
                
                #result_array = tensor2array(output,self.args.rgb_range)
                self.post_save_png_output(save_path,"Post_Output",nameext[0],post_image*255)
            # ---
            SORT_IQA_0_DIC = dict(sorted(IQA_0_DIC.items(), key=lambda item: item[1], reverse=True))
            for filename,iqa0 in SORT_IQA_0_DIC.items():
                info = "FileName: [{:}]  Dice: [{:.4f}]  MSE:[{:.4f}]  Softmiou:[{:.2f}] ".format(filename,IQA_0_DIC[filename],IQA_1_DIC[filename],IQA_2_DIC[filename])
                f.write(info+"\n")
            f.close()    
            # ---
            total_evaluation_time += evaluation_time
            self.ckp.log[-1, idx_data] /= len(d)
            AVGIQA1 = dataset_sum_iqa1/count
            AVGIQA2 = dataset_sum_iqa2/count
            self.evla_ckp_logwrite(d,idx_data,AVGIQA1,AVGIQA2)
            
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_evaluation_time,test_dataset_num / total_evaluation_time))
        self.ckp.write_log('Saving...')
        self.test_ckp_logend(timer_test.toc())
        torch.set_grad_enabled(True)

    def cat_mask_assement(self):
        torch.set_grad_enabled(False)
        self.evlaepoch_ckp_logstart()
        self.model_eval_control()
        timer_test = utility.timer()
        test_dataset_num ,total_evaluation_time = 0,0
        
        
        
        for idx_data, d in enumerate(self.loader_test):
            save_path = os.path.join('/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/experiment',
                                  self.args.pre_train, "Evluation","results-"+d.dataset.test_dataset_name,"Output")
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            test_dataset_num += len(d.dataset)
            evaluation_time = 0 # single dataset evaluation time
            AVGIQA1,AVGIQA2 = 0,0
            count =0
            dataset_sum_iqa1,dataset_sum_iqa2 = 0,0
            
            f = open(os.path.join('/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/experiment',
                                  self.args.pre_train, "Evluation","results-"+d.dataset.test_dataset_name,"iqa_log.txt"),'w')
            IQA_0_DIC = {}
            IQA_1_DIC = {}
            IQA_2_DIC = {}
            for image, ground_true, nameext in tqdm(d, ncols=80):
                image, ground_true = self.prepare(image, ground_true)
                #image_d ,ground_true = self.down_sample(image,ground_true)
                t_start = time.time()
                output = self.model_forward(image)
                output = self.output_process(output)
                if not self.args.cpu:
                    torch.cuda.synchronize()
                t_end = time.time()
                evaluation_time += t_end - t_start

                iqa0,iqa1,iqa2 = self.output_quality_eval(idx_data,output,ground_true)
                IQA_0_DIC[nameext[0]] = iqa0
                IQA_1_DIC[nameext[0]] = iqa1
                IQA_2_DIC[nameext[0]] = iqa2
                
                
                dataset_sum_iqa1 += iqa1
                dataset_sum_iqa2 += iqa2
                count +=1 
                
                
                output = biliary_mask(output,0.7)
                result_array = tensor2array(output,self.args.rgb_range)
                
                ground_true = tensor2arrayf(ground_true.cpu())
                
                image_array = tensor2arrayf(image.cpu())
                
                output_fusion = self.txt_name(show_mask_image(image_array*255,result_array),"output")
                output_fusion = self.txt_name(output_fusion,"Dice:"+str(iqa0)[:5],det=30)
                mask_fusion = self.txt_name(show_mask_image(image_array*255,ground_true*255),"label")
                
                cat_array = np.concatenate((mask_fusion,output_fusion),axis=1)
                
                
                self.post_save_png_output(save_path,"Mask_cat",nameext[0],cat_array)
            # ---
            SORT_IQA_0_DIC = dict(sorted(IQA_0_DIC.items(), key=lambda item: item[1], reverse=True))
            for filename,iqa0 in SORT_IQA_0_DIC.items():
                info = "FileName: [{:}]  Dice: [{:.4f}]  MSE:[{:.4f}]  Softmiou:[{:.2f}] ".format(filename,IQA_0_DIC[filename],IQA_1_DIC[filename],IQA_2_DIC[filename])
                f.write(info+"\n")
            f.close()

            # ---
            total_evaluation_time += evaluation_time
            self.ckp.log[-1, idx_data] /= len(d)
            AVGIQA1 = dataset_sum_iqa1/count
            AVGIQA2 = dataset_sum_iqa2/count
            self.evla_ckp_logwrite(d,idx_data,AVGIQA1,AVGIQA2)
            
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_evaluation_time,test_dataset_num / total_evaluation_time))
        self.ckp.write_log('Saving...')
        self.test_ckp_logend(timer_test.toc())
        torch.set_grad_enabled(True)

    def cat_post_process(self):
        torch.set_grad_enabled(False)
        self.evlaepoch_ckp_logstart()
        self.model_eval_control()
        timer_test = utility.timer()
        test_dataset_num ,total_evaluation_time = 0,0

        for idx_data, d in enumerate(self.loader_test):
            save_path = os.path.join('/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/experiment',
                                  self.args.pre_train, "Evluation","results-"+d.dataset.test_dataset_name,"Output")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            test_dataset_num += len(d.dataset)
            evaluation_time = 0 # single dataset evaluation time
            AVGIQA1,AVGIQA2 = 0,0
            count =0
            dataset_sum_iqa1,dataset_sum_iqa2 = 0,0
            
            f = open(os.path.join('/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/experiment',
                                  self.args.pre_train, "Evluation","results-"+d.dataset.test_dataset_name,"iqa_log.txt"),'w')
            IQA_0_DIC = {}
            IQA_1_DIC = {}
            IQA_2_DIC = {}
            for image, ground_true, nameext in tqdm(d, ncols=80):
                image, ground_true = self.prepare(image, ground_true)
                #image_d ,ground_true = self.down_sample(image,ground_true)
                t_start = time.time()
                output = self.model_forward(image)
                output = self.output_process(output)
                if not self.args.cpu:
                    torch.cuda.synchronize()
                t_end = time.time()
                evaluation_time += t_end - t_start

                iqa0,iqa1,iqa2 = self.output_quality_eval(idx_data,output,ground_true)
                IQA_0_DIC[nameext[0]] = iqa0
                IQA_1_DIC[nameext[0]] = iqa1
                IQA_2_DIC[nameext[0]] = iqa2
                
                
                dataset_sum_iqa1 += iqa1
                dataset_sum_iqa2 += iqa2
                count +=1 
                # ----- 
                mask_tensor = biliary_mask(output)
                mask_array = tensor2arrayf(mask_tensor.cpu())
                mask_array  = mcc_edge(mask_array*255)/255
                image_array = tensor2arrayf(image.cpu())
                
                post_image =postimage(image_array,mask_array)
                
                #post_image = self.txt_name(post_image,"output")
                #image_array = self.txt_name(image_array,"input")
                
                cat_image =  np.concatenate(( image_array,post_image),axis=1)
                #
                
                #result_array = tensor2array(output,self.args.rgb_range)
                self.post_save_png_output(save_path,"Cat_Post_Output_C08",nameext[0],cat_image*255)
            # ---
            SORT_IQA_0_DIC = dict(sorted(IQA_0_DIC.items(), key=lambda item: item[1], reverse=True))
            for filename,iqa0 in SORT_IQA_0_DIC.items():
                info = "FileName: [{:}]  Dice: [{:.4f}]  MSE:[{:.4f}]  Softmiou:[{:.2f}] ".format(filename,IQA_0_DIC[filename],IQA_1_DIC[filename],IQA_2_DIC[filename])
                f.write(info+"\n")
            f.close()    
            # ---
            total_evaluation_time += evaluation_time
            self.ckp.log[-1, idx_data] /= len(d)
            AVGIQA1 = dataset_sum_iqa1/count
            AVGIQA2 = dataset_sum_iqa2/count
            self.evla_ckp_logwrite(d,idx_data,AVGIQA1,AVGIQA2)
            
        self.ckp.write_log('\033[1;35m[ =======> Forward: {:.2f}s, FPS: {:.1f} <======= ] '.format(total_evaluation_time,test_dataset_num / total_evaluation_time))
        self.ckp.write_log('Saving...')
        self.test_ckp_logend(timer_test.toc())
        torch.set_grad_enabled(True)



    def txt_name(self,image,name,det=0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        info1 = name
        cv2.putText(image, info1, (10, 30+det), font, 0.8, (255, 255, 255), 2)
        return image




if __name__ == '__main__':
    from option import args
    args.data_test = ["N20Lumen"]
    args.data_train = ["N20Lumen"]
    args.model = "csafnet"
    args.pre_train = "[CSAF-EX11-Seg3-R2]-[N20Lumen_S11]-[2024-01-13-18-23]"
    args.resume = 0
    args.test_only =True
    args.cpu=True
    checkpoint = ckp.checkpoint_1model_1loss(args)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    #print(args.data_train)
    loader = data.Data(args)
    model = model.Model(args, checkpoint) 
    print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (sum(p.numel() for p in model.parameters())/1000000.0)) 
    assessment = Assessmen(args=args,loader=loader,model_list=[model],loss_list=[loss],ckp=checkpoint)
    #assessment.post_process()
    #assessment.assement()
    assessment.cat_mask_assement()
    assessment.cat_post_process()