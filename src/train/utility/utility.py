import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
from skimage.future import predict_segmenter
from torchmetrics.functional import accuracy

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
set_font = fm.FontProperties(fname="/home/ubuntu4090/下载/HarmonyOS Sans /HarmonyOS_Sans_SC/HarmonyOS_SansSC_Medium.ttf")


import numpy as np
import imageio
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import cv2
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from scipy import ndimage

from scipy.ndimage import binary_fill_holes

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
import seaborn as sns
from itertools import cycle




from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()
    def tic(self):
        self.t0 = time.time()
    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff
    def hold(self):
        self.acc += self.toc()
    def release(self):
        ret = self.acc
        self.acc = 0
        return ret
    def reset(self):
        self.acc = 0
        
        


# ----Function1: normalized the img
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_skit_psnr(img1,img2,rgb_range):
    img1 = np.array(img1.cpu())
    img2 = np.array(img2.cpu())
    psnr = peak_signal_noise_ratio(img1,img2,data_range=rgb_range)
    return psnr
    
def calc_skit_ssim(img1,img2,rgb_range):
    img1 = np.array(img1.cpu().squeeze(0).squeeze(0))
    img2 = np.array(img2.cpu().squeeze(0).squeeze(0))
    ssim = structural_similarity(img1,img2,data_range=rgb_range)
    return ssim

def calc_skit_mse(ten1,ten2):
    img1 = np.array(ten1.cpu())
    img2 = np.array(ten2.cpu())
    mse = mean_squared_error(img1,img2)
    return mse

def calc_accuray(prep,label,true_num,total_num):
    #print(prep.shape)
    _, predicted = torch.max(prep, 1)
    #print(predicted.shape)
    label_idx = label
    #print(predicted, "predicted" ,label , "label")
    correct = (predicted == label_idx).sum().item()
    #print(correct)
    if correct == 1:
        true_num = true_num + 1
        total_num = total_num + 1
    else:
        total_num = total_num +1

    accuracy = true_num / total_num
    return  accuracy,true_num,total_num

def calc_dice(img1, img2):
    """
    Calculate dice coefficient between two images.
    input (h,w) 255

    计算时候已经经过二值化处理了
    """
    img1 = np.array(img1.cpu())
    img2 = np.array(img2.cpu())

    img1 = np.round(img1).astype('uint8')
    img2 = np.round(img2).astype('uint8')
    s1 = np.sum(img1)
    s2 = np.sum(img2)
    s = np.sum(cv2.bitwise_and(img1, img2))
    d = 2 * s / (s1 + s2)
    return d

def calc_softmiou(pred, target):
    #输入都是C，H，W
    print(pred.shape)
    print(target.shape)


    pred = pred.squeeze(0)    # [c,h,w]
    target = target.squeeze(0)
    num_class = pred.shape[0]  # num_class = c
    def single_channel_to_onehot(label):
        # 验证输入图像是否为单通道
        if len(label.shape) != 2:
            raise ValueError("输入图像必须是单通道")
        # 获取图像尺寸
        h, w = label.shape
        # 初始化one-hot矩阵（全0）
        onehot = torch.zeros([num_class,h, w,]).to(torch.int8)
        # 为每个类别创建通道
        for class_id in range(0, num_class):  # 类别从1开始 # 这里忽略背景类别了
            # 创建当前类别的掩码
            mask = (label== class_id)
            # 在对应通道上标记
            onehot[class_id ,:, :] = mask#.to(torch.int8)  # 通道索引从0开始

        onehot_tensor = onehot
        return onehot_tensor

    # Old One
    # 都接受onehot 输入，也就是多个类别的多通道二值化 这个是多类别的哦
    # pred 中都是概率这里将转换成sinle 在变换成onehot
    single_output  = pred.argmax(dim=0)    # [h,w]
    onehot_output = single_channel_to_onehot(single_output)   # [c,h,w] [ 0,1 ]

    miou=torch.zeros([num_class])

    for i in range(num_class):
        output = onehot_output[i].cuda()
        targ = target[i].cuda()
        smooth = 1
        intersection = output * targ  # 这是什么意思，主要是分割中的软iou损失  ，
        iou = (intersection.sum() + smooth) / (output.sum() + targ.sum() - intersection.sum() + smooth)
        print(iou)
        #print(loss.device)
        miou[i]=iou

    loss = torch.mean(miou).cpu().item()
    return loss

def calc_multi_class_miou(pred_prob: torch.Tensor, true_onehot: torch.Tensor, threshold=None):
    """
       从概率图计算 mIoU
       :param pred_prob: [B, C, H, W] 概率输出
       :param true_onehot: [B, H, W] 真实标签（one-hot）
       :param threshold: 可选置信度阈值
       """
    B, C, H, W = pred_prob.shape
    pred_labels = torch.argmax(pred_prob, dim=1)  # [B, H, W]

    # 应用阈值过滤（可选）
    if threshold is not None:
        max_prob, _ = torch.max(pred_prob, dim=1)
        uncertain = (max_prob < threshold)
        pred_labels[uncertain] = C  # 标记为无效类别（不参与计算）

    # 生成预测的 one-hot 编码
    pred_onehot = torch.nn.functional.one_hot(pred_labels, C + 1)[..., :C]  # 移除无效类别
    pred_onehot = pred_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

    # 计算混淆矩阵（批量处理）
    confusion = torch.zeros(B, C, C, device=pred_prob.device)
    for b in range(B):
        for i in range(C):
            for j in range(C):
                # 统计真实为 i 且预测为 j 的像素数
                confusion[b, i, j] = torch.sum((true_onehot[b, i] == 1) & (pred_onehot[b, j] == 1))

    # 计算 TP, FP, FN
    tp = torch.diagonal(confusion, dim1=1, dim2=2)  # [B, C]
    fp = torch.sum(confusion, dim=1) - tp  # [B, C]
    fn = torch.sum(confusion, dim=2) - tp  # [B, C]
    union = tp + fp + fn

    # 计算各类别 IoU
    iou_per_class = tp / (union + 1e-8)  # [B, C]
    miou = torch.mean(iou_per_class[:,1:], dim=1).item()  # 每张图的平均 mIoU [B]

    return iou_per_class, miou



def calc_contrast(img1):
    #img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #彩色转为灰度图片
    m, n = img1.shape
    #图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0   # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 +
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)
    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式
    return cg


def calc_grad(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # 计算垂直方向梯度
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度的幅值和方向
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 计算平均梯度
    average_gradient = np.mean(gradient_magnitude)
    return average_gradient

### 这是用于图像分类的评价指标



class ClassificationEvaluator:
    def __init__(self, num_classes=10, class_names=None):
        """
        初始化评价器

        参数:
            num_classes (int): 类别数量
            class_names (list): 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f'Class {i}' for i in range(num_classes)]
        self.reset()

    def reset(self):
        """重置所有累积的统计量"""
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []

    def update(self, onehot_labels, pred_probs):
        """
        更新统计量

        参数:
            onehot_labels (np.ndarray): 真实标签p编码 (shape: [batch_size])
            pred_probs (np.ndarray): 预测概率 (shape: [batch_size, num_classes])
        """
        # 将one-hot标签转换为类别索引
        labels = np.array(onehot_labels.cpu())
        # 从预测概率获取预测类别
        preds = np.argmax(np.array(pred_probs.cpu()),axis=1)
        self.all_labels.append(labels)
        self.all_preds.append(preds)
        self.all_probs.append(np.array(pred_probs.cpu()))
    def compute_metrics(self):
        """计算所有评价指标"""
        labels = np.concatenate(self.all_labels)
        preds = np.concatenate(self.all_preds)
        probs = np.concatenate(self.all_probs)

        # 计算每个样本的准确度
        accuracy_per_sample = (preds == labels).astype(float)
        # 计算每个类的准确度
        class_acc = []
        for i in range(self.num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                class_acc.append(accuracy_per_sample[mask].mean())
            else:
                class_acc.append(float('nan'))

        # 计算精确度、召回率和F1分数
        precision = precision_score(labels, preds, average=None, zero_division=0)
        recall = recall_score(labels, preds, average=None, zero_division=0)
        f1 = f1_score(labels, preds, average=None, zero_division=0)

        # 计算混淆矩阵
        cm = confusion_matrix(labels, preds)

        # 计算PR曲线数据
        y_true_bin = np.concatenate([label for label in self.all_labels])
        y_true_bin = label_binarize(y_true_bin, classes=range(self.num_classes))
        precision_curve = dict()
        recall_curve = dict()
        average_precision = dict()

        for i in range(self.num_classes):
            precision_curve[i], recall_curve[i], _ = precision_recall_curve(
                y_true_bin[:, i], probs[:, i])
            average_precision[i] = average_precision_score(
                y_true_bin[:, i], probs[:, i])

        # 计算微平均PR曲线
        precision_curve["micro"], recall_curve["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), probs.ravel())
        average_precision["micro"] = average_precision_score(
            y_true_bin, probs, average="micro")

        # 计算宏平均PR曲线
        precision_curve["macro"] = np.linspace(0, 1, 100)
        recall_curve["macro"] = np.zeros_like(precision_curve["macro"])

        for i in range(self.num_classes):
            recall_curve["macro"] += np.interp(
                precision_curve["macro"],
                precision_curve[i][::-1],  # 反转以确保x是递增的
                recall_curve[i][::-1]
            )
        recall_curve["macro"] /= self.num_classes
        average_precision["macro"] = average_precision_score(
            y_true_bin, probs, average="macro")

        metrics = {
            'class_accuracy': class_acc,
            'mean_accuracy': np.nanmean(class_acc),
            'class_precision': precision,
            'mean_precision': precision.mean(),
            'class_recall': recall,
            'mean_recall': recall.mean(),
            'class_f1': f1,
            'mean_f1': f1.mean(),
            'confusion_matrix': cm,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'average_precision': average_precision
        }

        return metrics

    def plot_confusion_matrix(self, cm, title='Confusion matrix',save_path=None):
        """绘制混淆矩阵（始终显示具体样本数量）"""
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d',  # 始终使用整数格式
                    cmap='Blues', xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(title,fontproperties=set_font)
        plt.ylabel('True label',fontproperties=set_font)
        plt.xlabel('Predicted label',fontproperties=set_font)
        plt.xticks(rotation=90,fontproperties=set_font)
        plt.yticks(rotation=0,fontproperties=set_font)
        plt.tight_layout()
        fig.savefig(os.path.join(save_path,"confusion_matrix.png"))


    def plot_individual_pr_curves(self, precision_curve, recall_curve, average_precision,save_path=None):
        """绘制每个类的PR曲线"""
        fig = plt.figure(figsize=(10, 8))

        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal',
                        'red', 'green', 'blue', 'purple', 'brown'])

        for i, color in zip(range(self.num_classes), colors):
            plt.plot(recall_curve[i], precision_curve[i], color=color, lw=1,
                     label=f'{self.class_names[i]} (AP={average_precision[i]:0.2f})')

        plt.xlabel('Recall',fontproperties=set_font)
        plt.ylabel('Precision',fontproperties=set_font)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Per-Class Precision-Recall Curves',fontproperties=set_font)
        plt.legend(loc="best",prop=set_font)
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, "individual_pr_curves.png"))
        plt.show()

    def plot_average_pr_curves(self, precision_curve, recall_curve, average_precision,save_path=None):
        """绘制平均PR曲线"""
        fig = plt.figure(figsize=(10, 8))

        # 绘制微平均PR曲线
        plt.plot(recall_curve["micro"], precision_curve["micro"],
                 color='deeppink', linestyle=':', linewidth=4,
                 label=f'Micro-average (AP={average_precision["micro"]:0.2f})')

        # 绘制宏平均PR曲线
        plt.plot(recall_curve["macro"], precision_curve["macro"],
                 color='darkred', linestyle=':', linewidth=4,
                 label=f'Macro-average (AP={average_precision["macro"]:0.2f})')

        plt.xlabel('Recall',fontproperties=set_font)
        plt.ylabel('Precision',fontproperties=set_font)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Average Precision-Recall Curves',fontproperties=set_font)
        plt.legend(loc="best",prop=set_font)
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, "average_pr_curves.png"))
        plt.show()

    def print_metrics_report(self, metrics):
        """打印详细的指标报告"""
        print("\nClassification Metrics Report")
        print("=" * 50)

        # 打印每个类的指标
        print("\nPer-Class Metrics:")
        print(f"{'Class':<15}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'AP':<10}")
        for i in range(self.num_classes):
            print(f"{self.class_names[i]:<15}"
                  f"{metrics['class_accuracy'][i]:<10.4f}"
                  f"{metrics['class_precision'][i]:<10.4f}"
                  f"{metrics['class_recall'][i]:<10.4f}"
                  f"{metrics['class_f1'][i]:<10.4f}"
                  f"{metrics['average_precision'][i]:<10.4f}")

        # 打印平均指标
        print("\nAverage Metrics:")
        print(f"{'Mean Accuracy':<20}{metrics['mean_accuracy']:.4f}")
        print(f"{'Mean Precision':<20}{metrics['mean_precision']:.4f}")
        print(f"{'Mean Recall':<20}{metrics['mean_recall']:.4f}")
        print(f"{'Mean F1-Score':<20}{metrics['mean_f1']:.4f}")
        print(f"{'Micro AP':<20}{metrics['average_precision']['micro']:.4f}")
        print(f"{'Macro AP':<20}{metrics['average_precision']['macro']:.4f}")
    def information(self,metrics):
        """输出详细的指标报告"""
        information = "\nClassification Metrics Report"+"\n"
        information += "=" * 50+"\n"

        # 打印每个类的指标
        information +="\nPer-Class Metrics:"+"\n"
        information +=f"{'Class':<15}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'AP':<10}"+"\n"
        for i in range(self.num_classes):
            information +=f"{self.class_names[i]:<15}"+ \
                           f"{metrics['class_accuracy'][i]:<10.4f}"+\
                           f"{metrics['class_precision'][i]:<10.4f}"+\
                           f"{metrics['class_recall'][i]:<10.4f}"+\
                           f"{metrics['class_f1'][i]:<10.4f}"+\
                           f"{metrics['average_precision'][i]:<10.4f}"+"\n"

        # 打印平均指标
        information +="\nAverage Metrics:"+"\n"
        information +=f"{'Mean Accuracy':<20}{metrics['mean_accuracy']:.4f}"+"\n"
        information +=f"{'Mean Precision':<20}{metrics['mean_precision']:.4f}"+"\n"
        information +=f"{'Mean Recall':<20}{metrics['mean_recall']:.4f}"+"\n"
        information +=f"{'Mean F1-Score':<20}{metrics['mean_f1']:.4f}"+"\n"
        information +=f"{'Micro AP':<20}{metrics['average_precision']['micro']:.4f}"+"\n"
        information +=f"{'Macro AP':<20}{metrics['average_precision']['macro']:.4f}"+"\n"

        return information

    def evaluate(self,save_path):
        """执行完整评估并显示结果"""
        metrics = self.compute_metrics()

        # 打印报告
        self.print_metrics_report(metrics)

        # 绘制混淆矩阵（始终显示具体样本数量）
        self.plot_confusion_matrix(metrics['confusion_matrix'],save_path=save_path)

        # 绘制PR曲线（分为两个图）
        self.plot_individual_pr_curves(metrics['precision_curve'],
                                       metrics['recall_curve'],
                                       metrics['average_precision'],save_path=save_path)

        self.plot_average_pr_curves(metrics['precision_curve'],
                                    metrics['recall_curve'],
                                    metrics['average_precision'],save_path=save_path)
        information = self.information(metrics)
        return metrics,information




# class SegmentationMetric(object):
#     # 多类别语义分割衡量指标 ，输入预测图 前提需要argmax 处理 B，H，W 标签  B，H，W
#     def __init__(self, numClass,class_names):
#         self.numClass = numClass
#         self.confusionMatrix = np.zeros((self.numClass,) * 2)
#         self.segment_class_name = class_names
#
#     def genConfusionMatrix(self, imgPredict, imgLabel):
#         mask = (imgLabel >= 0) & (imgLabel < self.numClass)
#
#         label = self.numClass * imgLabel[mask] + imgPredict[mask]    # 这个 ·是一个编码方法 用于快速的计算每一种情况，实际上的情况就是将混淆矩阵的位置 从第一行第一列转换到 0到n*n 的编码
#         count = np.bincount(label, minlength=self.numClass ** 2)
#         confusionMatrix = count.reshape(self.numClass, self.numClass)
#         return confusionMatrix
#
#     def addBatch(self, imgPredict, imgLabel):
#         assert imgPredict.shape == imgLabel.shape
#         self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 更新一次混淆矩阵
#         return self.confusionMatrix
#
#     def pixelAccuracy(self):
#         acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
#         # 平均精度 混淆矩阵 对角线 就是每一类的正确数量
#         return acc
#
#     def classPixelAccuracy(self):
#         denominator = self.confusionMatrix.sum(axis=1)    # 计算 每一类的真实样本的所有数量
#         denominator = np.where(denominator == 0, 1e-12, denominator)
#         classAcc = np.diag(self.confusionMatrix) / denominator   # 这里的返回每一类的精确度
#         return classAcc
#
#     def meanPixelAccuracy(self):
#         classAcc = self.classPixelAccuracy()
#         meanAcc = np.nanmean(classAcc)
#         return meanAcc
#         # 这里是 每类的平均数据
#     def IntersectionOverUnion(self):
#         intersection = np.diag(self.confusionMatrix)   # 混淆矩阵的 对角就是 TP
#         union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
#             self.confusionMatrix)
#         # axis =1 是对 列维度压缩 ，计算的是实际每一类的所有数量
#
#         # 这里 的np .sum 后面的 axis 假设原来的H，W  则 sum (axis = 0） 结果是 [1,w] 对行维度压缩，计算的是每一列的和  则 sum (axis = 1） 结果是 [H,1] 对列 维度压缩，计算的是每行的和
#         union = np.where(union == 0, 1e-12, union)
#         IoU = intersection / union
#         return IoU    # 这里面返回每一个类的IOU  [N]
#
#     def meanIntersectionOverUnion(self):
#         mIoU = np.nanmean(self.IntersectionOverUnion())   # 返回的 平均 IOU
#         return mIoU
#
#     def Frequency_Weighted_Intersection_over_Union(self):
#         denominator1 = np.sum(self.confusionMatrix)  # 全局的所有的像素数量
#         denominator1 = np.where(denominator1 == 0, 1e-12, denominator1)
#         # 这是计算
#         freq = np.sum(self.confusionMatrix, axis=1) / denominator1   # 这是 计算每一个 实际 出现的频率
#         # 这是 并级
#         denominator2 = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
#             self.confusionMatrix)
#
#         denominator2 = np.where(denominator2 == 0, 1e-12, denominator2)
#         iu = np.diag(self.confusionMatrix) / denominator2
#         FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()  # 频率乘之前的iou 在 求和
#         return FWIoU
#
#     def classF1Score(self):
#         tp = np.diag(self.confusionMatrix)
#         fp = self.confusionMatrix.sum(axis=0) - tp  # 对于预测的每一类的数量 减 正确的数量 剩下的就是被错误预测为当前类的数量
#         fn = self.confusionMatrix.sum(axis=1) - tp   # 对与所有的实际数量 减去正确的数量 就是当前类被错误预测为 其他类的数量
#
#         precision = tp / (tp + fp + 1e-12)
#         recall = tp / (tp + fn + 1e-12)
#
#         f1 = 2 * precision * recall / (precision + recall + 1e-12)
#         return f1   # 计算好精确度和召回之后
#
#     def meanF1Score(self):
#         f1 = self.classF1Score()
#         mean_f1 = np.nanmean(f1)
#         return mean_f1
#
#     def reset(self):
#         self.confusionMatrix = np.zeros((self.numClass, self.numClass))
#
#     def get_scores(self):
#         scores = {
#             'Pixel Accuracy': self.pixelAccuracy(),
#             'Class Pixel Accuracy': self.classPixelAccuracy(),
#             'Intersection over Union': self.IntersectionOverUnion(),
#             'Class F1 Score': self.classF1Score(),
#             'Frequency Weighted Intersection over Union': self.Frequency_Weighted_Intersection_over_Union(),
#             'Mean Pixel Accuracy': self.meanPixelAccuracy(),
#             'Mean Intersection over Union(mIoU)': self.meanIntersectionOverUnion(),
#             'Mean F1 Score': self.meanF1Score()
#         }
#         return scores
#     def get_information(self):
#         scores = self.get_scores()
#         information = "-"*20+"Segment Evuation: "+"-"*20+'\n'
#         key_list = []
#         value_list = []
#         for k, v in scores.items():
#             if isinstance(v, np.ndarray):
#                 # information += f"{k:<20}: {np.round(v, 3)}"+'\n'
#                 key_list.append(k)
#                 value_list.append(v)
#             else:
#                 information += f"{k}: {v:.4f}"+'\n'
#
#         information += f"{'组织类别':<30}"
#         for i in range(len(key_list)):
#             information += f"{key_list[i]:<20}"
#         information+='\n'
#         for seg_idx in range(len( self.segment_class_name)) :
#             information += f"{self.segment_class_name[seg_idx]:<30}  {value_list[0][seg_idx]:<20.4f}  {value_list[1][seg_idx]:<20.4f}  {value_list[2][seg_idx]:<20.4f} "  +'\n'
#
#         return  information
#
import numpy as np

class SegmentationMetric(object):
    # 多类别语义分割衡量指标 ，输入预测图 前提需要argmax 处理 B，H，W 标签  B，H，W
    def __init__(self, numClass, class_names):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass, self.numClass), dtype=np.float64)
        self.segment_class_name = class_names

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype(np.int64) + imgPredict[mask].astype(np.int64)
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass).astype(np.float64)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 更新一次混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass), dtype=np.float64)

    # 基本项：TP, FP, FN
    def _tp_fp_fn(self):
        tp = np.diag(self.confusionMatrix).astype(np.float64)
        fp = self.confusionMatrix.sum(axis=0) - tp  # predicted as class j but true is other
        fn = self.confusionMatrix.sum(axis=1) - tp  # true class i but predicted as other
        return tp, fp, fn

    def pixelAccuracy(self):
        total = self.confusionMatrix.sum()
        total = total if total != 0 else 1e-12
        acc = np.diag(self.confusionMatrix).sum() / total
        return acc

    def classPixelAccuracy(self):
        denominator = self.confusionMatrix.sum(axis=1)    # 每一类的真实样本数
        denominator = np.where(denominator == 0, 1e-12, denominator)
        classAcc = np.diag(self.confusionMatrix) / denominator
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - intersection
        union = np.where(union == 0, 1e-12, union)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        total = np.sum(self.confusionMatrix)
        total = total if total != 0 else 1e-12
        freq = np.sum(self.confusionMatrix, axis=1) / total
        denom2 = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        denom2 = np.where(denom2 == 0, 1e-12, denom2)
        iu = np.diag(self.confusionMatrix) / denom2
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # precision, recall, f1 (per-class)
    def classPrecision(self):
        tp, fp, _ = self._tp_fp_fn()
        precision = tp / (tp + fp + 1e-12)
        return precision

    def meanPrecision(self):
        return np.nanmean(self.classPrecision())

    def classRecall(self):
        tp, _, fn = self._tp_fp_fn()
        recall = tp / (tp + fn + 1e-12)
        return recall

    def meanRecall(self):
        return np.nanmean(self.classRecall())

    def classF1Score(self):
        tp, fp, fn = self._tp_fp_fn()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return f1

    def meanF1Score(self):
        f1 = self.classF1Score()
        mean_f1 = np.nanmean(f1)
        return mean_f1

    # Dice (per-class) 与 mean Dice
    def classDice(self):
        tp, fp, fn = self._tp_fp_fn()
        # Dice = 2*TP / (2*TP + FP + FN)  等价于 2TP / (sum_row + sum_col)
        denom = 2 * tp + fp + fn
        denom = np.where(denom == 0, 1e-12, denom)
        dice = 2 * tp / denom
        return dice

    def meanDice(self):
        return np.nanmean(self.classDice())

    # 保持原有的 get_scores，但加入新指标（数组型指标仍作为 ndarray 返回）
    def get_scores(self):
        scores = {
            'Pixel Accuracy': self.pixelAccuracy(),
            'Class  Accuracy': self.classPixelAccuracy(),
            'Class Iou': self.IntersectionOverUnion(),
            'Class F1 Score': self.classF1Score(),
            'Frequency Weighted Intersection over Union': self.Frequency_Weighted_Intersection_over_Union(),
            'Mean Pixel Accuracy': self.meanPixelAccuracy(),
            'Mean Intersection over Union(mIoU)': self.meanIntersectionOverUnion(),
            'Mean F1 Score': self.meanF1Score(),
            # 新增
            'Class Precision': self.classPrecision(),
            'Mean Precision': self.meanPrecision(),
            'Class Recall': self.classRecall(),
            'Mean Recall': self.meanRecall(),
            'Class Dice': self.classDice(),
            'Mean Dice': self.meanDice()
        }
        return scores

    def get_information(self):
        scores = self.get_scores()
        information = "-"*10 + " Segment Evaluation " + "-"*10 + '\n'
        key_list = []
        value_list = []
        # 先把标量项写出，数组项收集起来以表格形式输出
        for k, v in scores.items():
            if isinstance(v, np.ndarray):
                key_list.append(k)
                value_list.append(v)
            else:
                information += f"{k}: {v:.4f}\n"

        # 如果没有数组型指标，直接返回
        if len(key_list) == 0:
            return information

        # 表头
        information += f"\n{'Class Name':<30}"
        for key in key_list:
            information += f"{key:<20}"
        information += '\n'

        # 每个类的行
        n = self.numClass #len(self.segment_class_name)
        # 保证 value_list 中每个都是长度为 numClass 的 ndarray
        for idx in range(n):
            information += f"{self.segment_class_name[idx]:<30}"
            for arr in value_list:
                val = arr[idx]
                information += f"{val:<20.4f}"
            information += '\n'


        return information

import torch
import cv2
import numpy as np
from skimage import measure
from scipy import ndimage

import torch
import cv2
import numpy as np
from skimage import measure
from scipy.interpolate import splprep, splev

class ArgmaxPostProcessor:
    def __init__(self, kernel_size=7, min_area=80,
                 background_class=0, smoothing_iter=2):
        """
        Argmax预测图后处理优化模块

        参数:
            kernel_size: 形态学操作核大小 (默认3x3)
            min_area: 最小连通区域面积阈值 (默认50像素)
            background_class: 背景类别索引 (默认0)
            smoothing_iter: 平滑迭代次数 (默认2)
        """
        self.kernel_size = kernel_size
        #self.kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))  # 闭运算是填充这个尺寸大一点
        #self.kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.min_area = min_area
        self.background_class = background_class
        self.smoothing_iter = smoothing_iter
        self.smoother = FrameSmoother(num_classes=24,window_size=9,device='cuda')


    def process_batch(self, pred_labels):
        """
        批量处理argmax预测图

        输入:
            pred_labels: 模型输出的类别标签图 (B,H,W) [0, num_classes-1]
        返回:
            processed: 后处理后的预测图 (B,H,W)
        """
        H, W = pred_labels.shape
        processed = torch.zeros_like(pred_labels)
        processed= self._process_single_image(pred_labels)
        return processed

    def _process_single_image(self, label_map):
        """
        处理单张预测图

        步骤:
        1. 分离非背景类别
        2. 对每个非背景类别进行形态学优化
        3. 合并处理后的类别
        """
        # 转换为NumPy数组
        labels = label_map.cpu().numpy().astype(np.uint8)   # P 模式的 1,H,W

        # 创建处理后的标签图
        processed_labels = np.zeros_like(labels)

        # 获取所有非背景类别
        non_bg_classes = np.unique(labels)
        non_bg_classes = non_bg_classes[non_bg_classes != self.background_class]

        # 处理每个非背景类别
        for cls in non_bg_classes:
            # 提取当前类别的二值掩码
            cls_mask = (labels == cls).astype(np.uint8)   # 实际上 已经 是 等价 与 概率图1

            # 形态学优化  需要用的
            if cls == 1:
                optimized_mask = self._optimize_class_mask(cls_mask,NMS=False)

            else :
                optimized_mask = self._optimize_class_mask(cls_mask)
            #optimized_mask = cls_mask
            # 将优化后的掩码添加到处理后的标签图
            processed_labels[optimized_mask > 0.2] = cls

        return torch.from_numpy(processed_labels)

    def _optimize_class_mask(self, mask,NMS = True):
        """
        优化单个类别的二值掩码
        步骤:
        1. 形态学开闭运算
        2. 连通区域分析
        3. 小区域过滤
        4. 边缘平滑
        """

        # 1. 连通区域分析
        labeled_mask, num_labels = measure.label(mask, connectivity=2, return_num=True)
        regions = measure.regionprops(labeled_mask)

        # 2. 小区域过滤  --->  只保留最大区域
        filtered_mask = np.zeros_like(mask)
        Max_area = 0
        Max_region = 0

        if NMS:
            for region in regions:
                if region.area >= Max_area:
                    Max_area = region.area
                    Max_region = region
                    coords = region.coords

            filtered_mask[coords[:, 0], coords[:, 1]] = 1
        else:
            for region in regions:
                if region.area >300:
                    coords = region.coords
                    filtered_mask[coords[:, 0], coords[:, 1]] = 1

                Max_region = region

        # 1. 形态学开闭运算
        # major_axis_length = Max_region.major_axis_length
        minor_axis_length = Max_region.minor_axis_length
        # print(minor_axis_length)
        if minor_axis_length < self.kernel_size or  minor_axis_length < self.kernel_size :

            self.kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(minor_axis_length / 3),5),max(int(minor_axis_length / 3),5)))
            self.kernel1  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(minor_axis_length / 6),5), max(int(minor_axis_length / 6),5)))

        else:
            self.kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
            self.kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        for _ in range(self.smoothing_iter):
            # 闭运算填充空洞
            filtered_mask  = cv2.morphologyEx(filtered_mask , cv2.MORPH_OPEN, self.kernel2)

            filtered_mask  = cv2.morphologyEx(filtered_mask , cv2.MORPH_CLOSE, self.kernel1)
            # 开运算去除毛刺

            # 填补空洞
        filtered_mask = binary_fill_holes(filtered_mask).astype(np.uint8) * 255

        smoothed_mask = cv2.GaussianBlur(filtered_mask.astype(np.float32), (9, 9), 0.5)
        #smoothed_mask = (smoothed_mask > 0.3).astype(np.uint8)

        return smoothed_mask

    def _optimize_class_maskv2(self, mask,NMS = True,spline_smooth_factor=3.0, num_points=300):
        """
        优化单个类别的二值掩码
        步骤:
        1. 形态学开闭运算
        2. 连通区域分析
        3. 小区域过滤
        4. 边缘轮廓样条平滑
        """

        # 1. 连通区域分析
        labeled_mask, num_labels = measure.label(mask, connectivity=2, return_num=True)
        regions = measure.regionprops(labeled_mask)

        # 2. 小区域过滤  --->  只保留最大区域
        filtered_mask = np.zeros_like(mask)
        Max_area = 0
        Max_region = 0

        if NMS:
            for region in regions:
                if region.area >= Max_area:
                    Max_area = region.area
                    Max_region = region
                    coords = region.coords

            filtered_mask[coords[:, 0], coords[:, 1]] = 1
        else:
            for region in regions:
                if region.area >300:
                    coords = region.coords
                    filtered_mask[coords[:, 0], coords[:, 1]] = 1

                Max_region = region

        # 1. 形态学开闭运算
        # major_axis_length = Max_region.major_axis_length
        minor_axis_length = Max_region.minor_axis_length
        # print(minor_axis_length)
        if minor_axis_length < self.kernel_size or  minor_axis_length < self.kernel_size :

            self.kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(minor_axis_length / 3),3),max(int(minor_axis_length / 3),3)))
            self.kernel1  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(minor_axis_length / 6),3), max(int(minor_axis_length / 6),3)))

        else:
            self.kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
            self.kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3 ))

        for _ in range(self.smoothing_iter):
            # 闭运算填充空洞
            filtered_mask  = cv2.morphologyEx(filtered_mask , cv2.MORPH_OPEN, self.kernel2)

            filtered_mask  = cv2.morphologyEx(filtered_mask , cv2.MORPH_CLOSE, self.kernel1)
            # 开运算去除毛刺

            # 填补空洞
        filtered_mask = binary_fill_holes(filtered_mask).astype(np.uint8) * 255

        # ---------- 3. 边界提取 ----------
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return filtered_mask

        contour = max(contours, key=cv2.contourArea)
        contour = contour.squeeze(1)  # (N, 2)

        # 太小的区域跳过
        #if contour.shape[0] < 20:
        #    return filtered_mask

        # ---------- 4. B样条平滑 ----------
        x, y = contour[:, 0], contour[:, 1]

        # 闭合轮廓
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

        # 使用B样条拟合 (周期性)
        try:
            tck, u = splprep([x, y], s=spline_smooth_factor, per=True)
            unew = np.linspace(0, 1, num_points)
            x_smooth, y_smooth = splev(unew, tck)
        except Exception as e:
            print(f"[WARN] B-spline smoothing failed: {e}")
            return filtered_mask

        # ---------- 5. 重建平滑掩码 ----------
        smooth_contour = np.stack([x_smooth, y_smooth], axis=1).astype(np.int32)
        smoothed_mask = np.zeros_like(filtered_mask)
        cv2.fillPoly(smoothed_mask, [smooth_contour], 1)
        smoothed_mask = cv2.GaussianBlur( smoothed_mask.astype(np.float32), (9, 9), 0.5)

        return smoothed_mask

    @staticmethod
    def onehot_to_mask(onehot: torch.Tensor) -> torch.Tensor:
        """
        将 one-hot 转换为类别 mask (P模式).
        输入: [N, C, H, W]
        输出: [N, H, W] (long类型, 每个像素是类别id)
        """
        return onehot.argmax(dim=1).long()

    @staticmethod
    def mask_to_onehot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
        N, H, W = mask.shape
        onehot = torch.zeros((N, num_classes, H, W), device=mask.device, dtype=torch.float)
        onehot.scatter_(1, mask.unsqueeze(1), 1.0)
        return onehot

    @staticmethod
    def compute_class_distribution(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        统计 mask 中的类分布 (类别是否出现).
        输入: [N, H, W]
        输出: [N, num_classes] (0/1，表示该类是否出现)
        """
        N = mask.shape[0]
        dist = torch.zeros((N, num_classes), device=mask.device, dtype=torch.int)
        for n in range(N):
            unique_labels = torch.unique(mask[n])
            dist[n, unique_labels] = 1
        return dist

    @staticmethod
    def morph_erode(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        腐蚀操作 (二值 mask)
        输入: [N,1,H,W] float
        输出: [N,1,H,W] float
        """
        pad = kernel_size // 2
        eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
        return eroded

    @staticmethod
    def compute_iou(mask1: torch.Tensor, mask2: torch.Tensor, num_classes: int) -> float:
        """
        计算两个 mask 的 mean IoU
        输入: [H, W] x2
        输出: scalar (float)
        """
        ious = []
        for cls in range(num_classes):
            inter = torch.logical_and(mask1 == cls, mask2 == cls).sum().item()
            union = torch.logical_or(mask1 == cls, mask2 == cls).sum().item()
            if union > 0:
                ious.append(inter / union)
        if len(ious) == 0:
            return 0.0
        return sum(ious) / len(ious)


    def temporal_mask_smoothing(self,prev_onehot: torch.Tensor,
                                curr_onehot: torch.Tensor,
                                num_classes: int,
                                kernel_size: int = 5) -> torch.Tensor:
        """
        帧间平滑 (只对前景类进行缓冲，背景类 idx=0 不变)

        prev_onehot: 上一帧预测 [N,C,H,W]
        curr_onehot: 当前帧预测 [N,C,H,W]
        num_classes: 类别数
        kernel_size: 缓冲强度 (越大，边界移动越慢)

        返回: 平滑后的 one-hot [N,C,H,W]
        """
        prev_mask = prev_onehot.argmax(dim=1)  # [N,H,W]
        curr_mask = curr_onehot.argmax(dim=1)

        N, H, W = prev_mask.shape
        smoothed_mask = curr_mask.clone()

        for n in range(N):
            for cls in range(1, num_classes):  # ⚠️ 跳过背景类 0
                prev_region = (prev_mask[n] == cls).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                curr_region = (curr_mask[n] == cls).float().unsqueeze(0).unsqueeze(0)

                added = torch.clamp(curr_region - prev_region, 0, 1)  # 新增区域
                removed = torch.clamp(prev_region - curr_region, 0, 1)  # 消失区域

                # 腐蚀：只保留边界部分
                added_eroded = self.morph_erode(added, kernel_size)
                removed_eroded = self.morph_erode(removed, kernel_size)

                # 最终区域 = 原区域 + 缓冲新增 - 缓冲消失
                final_region = ((prev_region + added_eroded - removed_eroded) > 0.5).squeeze().long()

                smoothed_mask[n][final_region == 1] = cls

        return self.mask_to_onehot(smoothed_mask, num_classes)

    def temporal_smoothing(self,prev_onehot: torch.Tensor,
                           curr_onehot: torch.Tensor,
                           num_classes: int,
                           iou_thresh: float = 0.7,
                           alpha: float = 0.5) -> torch.Tensor:
        """
        帧间平滑处理
        prev_onehot: 上一帧预测 (one-hot, [N, C, H, W])
        curr_onehot: 当前帧预测 (one-hot, [N, C, H, W])
        num_classes: 类别数
        iou_thresh: IoU 阈值
        alpha: 平滑系数 (0~1)

        返回: 平滑后的 one-hot 输出 [N, C, H, W]
        """
        prev_mask = self.onehot_to_mask(prev_onehot)
        curr_mask = self.onehot_to_mask(curr_onehot)

        prev_dist = self.compute_class_distribution(prev_mask, num_classes)
        curr_dist = self.compute_class_distribution(curr_mask, num_classes)

        smoothed = curr_onehot.clone()

        for n in range(prev_onehot.shape[0]):
            if torch.equal(prev_dist[n], curr_dist[n]):  # 类分布一致
                iou = self.compute_iou(prev_mask[n], curr_mask[n], num_classes)

                if iou >= iou_thresh:
                    # 帧间平滑: 对 one-hot 概率加权平均
                    smoothed[n] = F.normalize(
                        alpha * prev_onehot[n] + (1 - alpha) * curr_onehot[n],
                        p=1, dim=0
                    )
                    #smoothed=self.temporal_mask_smoothing(prev_onehot,curr_onehot,num_classes,kernel_size=7)

        return smoothed

    def temporal_smoothing_v2(self,prev_onehot: torch.Tensor,
                           curr_onehot: torch.Tensor,
                           num_classes: int) -> torch.Tensor:
        """
        帧间平滑处理
        prev_onehot: 上一帧预测 (one-hot, [N, C, H, W]) 这里是 softmax 以及 使用的是多帧的滑窗
        curr_onehot: 当前帧预测 (one-hot, [N, C, H, W])
        num_classes: 类别数
        iou_thresh: IoU 阈值
        alpha: 平滑系数 (0~1)

        返回: 平滑后的 softmax 输出 [N, C, H, W]
        """
        prev_mask = self.onehot_to_mask(prev_onehot)
        curr_mask = self.onehot_to_mask(curr_onehot)

        prev_dist = self.compute_class_distribution(prev_mask, num_classes)
        curr_dist = self.compute_class_distribution(curr_mask, num_classes)

        smoothed = curr_onehot.clone()

        for n in range(prev_onehot.shape[0]):
            if torch.equal(prev_dist[n], curr_dist[n]):  # 类分布一致   # 在我们测试数据集中  但是会出现一个同一类的不同case 中的混淆程度
                #iou = self.compute_iou(prev_mask[n], curr_mask[n], num_classes)
                #if iou >= iou_thresh:
                smoothed = self.smoother.smooth(curr_onehot)
                    #smoothed=self.temporal_mask_smoothing(prev_onehot,curr_onehot,num_classes,kernel_size=7)
                smoothed[smoothed<=(1/self.smoother.window_size)]==0
            else:
                self.smoother.reset()
        return smoothed


from collections import deque

class FrameSmoother:
    def __init__(self, num_classes=24, window_size=3, device="cuda"):
        """
        :param num_classes: 语义分割类别数
        :param window_size: 滑动窗口大小 (比如3~5，越大越平滑)
        """
        self.num_classes = num_classes
        self.window_size = window_size
        self.recent_probs = deque(maxlen=window_size)
        self.device = device

    def smooth(self, logits):
        """
        输入: 模型原始输出 (B, C, H, W)
        输出: 平滑后的概率 (B, C, H, W)
        """
        probs = F.softmax(logits, dim=1)  # 概率空间
        self.recent_probs.append(probs.detach())  # 保存历史帧
        probs_smooth = torch.mean(torch.stack(list(self.recent_probs), dim=0), dim=0)
        return probs_smooth

    def reset(self):
        self.recent_probs.clear()




import cv2
import numpy as np
from skimage import measure
from scipy import ndimage


class RobustSegPostProcessor:
    def __init__(self, kernel_size=8, min_area=30,
                 background_class=0, close_iter=3, open_iter=2,
                 gaussian_size=5, gaussian_sigma=1.5):
        """
        鲁棒的语义分割后处理器

        参数:
            kernel_size: 形态学操作核大小 (默认5x5)
            min_area: 最小连通区域面积阈值 (默认30像素)
            background_class: 背景类别索引 (默认0)
            close_iter: 闭运算迭代次数 (默认3)
            open_iter: 开运算迭代次数 (默认2)
            gaussian_size: 高斯模糊核大小 (默认5x5)
            gaussian_sigma: 高斯模糊标准差 (默认1.5)
        """
        # 创建不同大小的形态学核
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (max(3, kernel_size - 2), max(3, kernel_size - 2)))

        self.min_area = min_area
        self.background_class = background_class
        self.close_iter = close_iter
        self.open_iter = open_iter
        self.gaussian_size = gaussian_size
        self.gaussian_sigma = gaussian_sigma

    def process_batch(self, pred_labels):
        """
        批量处理argmax预测图

        输入:
            pred_labels: 模型输出的类别标签图 (B,H,W) [0, num_classes-1]
        返回:
            processed: 后处理后的预测图 (B,H,W)
        """
        H, W = pred_labels.shape
        processed = torch.zeros_like(pred_labels)
            # 处理单张预测图
        processed= self._process_single_image(pred_labels)

        return processed

    def _process_single_image(self, label_map):
        """
        处理单张预测图 - 增强内部填充和边缘平滑

        步骤:
        1. 分离非背景类别
        2. 对每个非背景类别进行形态学优化
        3. 合并处理后的类别
        """
        # 转换为NumPy数组
        labels = label_map.cpu().numpy().astype(np.uint8)

        # 创建处理后的标签图
        processed_labels = np.zeros_like(labels)

        # 获取所有非背景类别
        non_bg_classes = np.unique(labels)
        non_bg_classes = non_bg_classes[non_bg_classes != self.background_class]

        # 处理每个非背景类别
        for cls in non_bg_classes:
            # 提取当前类别的二值掩码
            cls_mask = (labels == cls).astype(np.uint8)

            # 增强形态学优化
            optimized_mask = self._enhanced_optimize_mask(cls_mask)

            # 将优化后的掩码添加到处理后的标签图
            # 确保维度匹配：使用相同的形状
            processed_labels = np.where(optimized_mask > 0, cls, processed_labels)

        return torch.from_numpy(processed_labels)

    def _enhanced_optimize_mask(self, mask):
        """
        增强型掩码优化 - 重点强化内部填充和边缘平滑

        步骤:
        1. 多阶段闭运算填充内部空洞
        2. 连通区域分析
        3. 小区域过滤
        4. 多阶段开运算平滑边缘
        5. 高级边缘平滑
        """
        # 1. 多阶段闭运算填充内部空洞
        for _ in range(self.close_iter):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        # 2. 连通区域分析
        labeled_mask, num_labels = measure.label(mask, connectivity=2, return_num=True)
        regions = measure.regionprops(labeled_mask)

        # 3. 小区域过滤
        filtered_mask = np.zeros_like(mask)
        for region in regions:
            if region.area >= self.min_area:
                # 使用凸包填充区域，确保内部完全填充
                hull_mask = self._convex_hull_fill(region, mask.shape)
                # 确保维度匹配：使用逻辑或操作
                filtered_mask = np.logical_or(filtered_mask, hull_mask).astype(np.uint8)

        # 4. 多阶段开运算平滑边缘
        for _ in range(self.open_iter):
            filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, self.kernel_open)

        # 5. 高级边缘平滑
        smoothed_mask = self._advanced_edge_smoothing(filtered_mask)

        return smoothed_mask

    def _convex_hull_fill(self, region, image_shape):
        """
        使用凸包填充区域，确保内部完全填充
        """
        # 获取区域坐标
        coords = region.coords

        # 计算凸包
        hull_points = cv2.convexHull(np.array(coords[:, [1, 0]]))  # 注意坐标顺序转换

        # 创建全图大小的掩码
        hull_mask = np.zeros(image_shape, dtype=np.uint8)

        # 填充凸包
        cv2.fillConvexPoly(hull_mask, hull_points, 1)

        return hull_mask

    def _advanced_edge_smoothing(self, mask):
        """
        高级边缘平滑技术
        """
        # 高斯模糊
        smoothed = cv2.GaussianBlur(mask.astype(np.float32),
                                    (self.gaussian_size, self.gaussian_size),
                                    self.gaussian_sigma)

        # 双边滤波 - 保留边缘的同时平滑
        smoothed = cv2.bilateralFilter(smoothed, d=5, sigmaColor=0.3, sigmaSpace=5)

        # 自适应阈值
        adaptive_thresh = cv2.adaptiveThreshold(
            (smoothed * 255).astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # 邻域大小
            2  # 常数C
        )

        return (adaptive_thresh > 0).astype(np.uint8)

    def visualize_comparison(self, original, processed, batch=0):
        """
        可视化原始预测和处理后的预测对比

        参数:
            original: 原始预测标签图 (B,H,W)
            processed: 处理后预测图 (B,H,W)
            batch: 要可视化的批次索引
        """
        import matplotlib.pyplot as plt

        # 提取原始和处理后的图像
        orig_img = original[batch].cpu().numpy()
        proc_img = processed[batch].cpu().numpy()

        # 创建对比图
        plt.figure(figsize=(20, 10))

        # 原始预测
        plt.subplot(231)
        plt.imshow(orig_img, cmap='jet')
        plt.title('Original Prediction')
        plt.colorbar()

        # 处理后预测
        plt.subplot(232)
        plt.imshow(proc_img, cmap='jet')
        plt.title('Processed Prediction')
        plt.colorbar()

        # 边缘对比
        plt.subplot(233)
        from skimage.segmentation import mark_boundaries
        orig_boundaries = mark_boundaries(np.zeros_like(orig_img), orig_img, color=(1, 0, 0))
        proc_boundaries = mark_boundaries(np.zeros_like(proc_img), proc_img, color=(0, 1, 0))
        plt.imshow(orig_boundaries)
        plt.imshow(proc_boundaries, alpha=0.5)
        plt.title('Edge Comparison (Red:Original, Green:Processed)')

        # 差异图
        plt.subplot(234)
        diff = np.abs(orig_img - proc_img)
        plt.imshow(diff, cmap='hot')
        plt.title('Difference')
        plt.colorbar()

        # 原始预测的边界
        plt.subplot(235)
        orig_edges = cv2.Canny((orig_img > 0).astype(np.uint8) * 255, 100, 200)
        plt.imshow(orig_edges, cmap='gray')
        plt.title('Original Edges')

        # 处理后预测的边界
        plt.subplot(236)
        proc_edges = cv2.Canny((proc_img > 0).astype(np.uint8) * 255, 100, 200)
        plt.imshow(proc_edges, cmap='gray')
        plt.title('Processed Edges')

        plt.tight_layout()
        plt.show()


# piqe 计算 
def calculate_mscn(dis_image):
    dis_image = dis_image.astype(np.float32)  # 类型转换十分重要
    ux = cv2.GaussianBlur(dis_image, (7, 7), 7/6)
    ux_sq = ux*ux
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(dis_image**2, (7, 7), 7/6)-ux_sq))

    mscn = (dis_image-ux)/(1+sigma)

    return mscn
# Function to segment block edges
def segmentEdge(blockEdge, nSegments, blockSize, windowSize):
    # Segment is defined as a collection of 6 contiguous pixels in a block edge
    segments = np.zeros((nSegments, windowSize))
    for i in range(nSegments):
        segments[i, :] = blockEdge[i:windowSize]
        if(windowSize <= (blockSize+1)):
            windowSize = windowSize+1

    return segments
def noticeDistCriterion(Block, nSegments, blockSize, windowSize, blockImpairedThreshold, N):
    # Top edge of block
    topEdge = Block[0, :]
    segTopEdge = segmentEdge(topEdge, nSegments, blockSize, windowSize)

    # Right side edge of block
    rightSideEdge = Block[:, N-1]
    rightSideEdge = np.transpose(rightSideEdge)
    segRightSideEdge = segmentEdge(
        rightSideEdge, nSegments, blockSize, windowSize)

    # Down side edge of block
    downSideEdge = Block[N-1, :]
    segDownSideEdge = segmentEdge(
        downSideEdge, nSegments, blockSize, windowSize)

    # Left side edge of block
    leftSideEdge = Block[:, 0]
    leftSideEdge = np.transpose(leftSideEdge)
    segLeftSideEdge = segmentEdge(
        leftSideEdge, nSegments, blockSize, windowSize)

    # Compute standard deviation of segments in left, right, top and down side edges of a block
    segTopEdge_stdDev = np.std(segTopEdge, axis=1)
    segRightSideEdge_stdDev = np.std(segRightSideEdge, axis=1)
    segDownSideEdge_stdDev = np.std(segDownSideEdge, axis=1)
    segLeftSideEdge_stdDev = np.std(segLeftSideEdge, axis=1)

    # Check for segment in block exhibits impairedness, if the standard deviation of the segment is less than blockImpairedThreshold.
    blockImpaired = 0
    for segIndex in range(segTopEdge.shape[0]):
        if((segTopEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segRightSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segDownSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segLeftSideEdge_stdDev[segIndex] < blockImpairedThreshold)):
            blockImpaired = 1
            break

    return blockImpaired
def noiseCriterion(Block, blockSize, blockVar):
    # Compute block standard deviation[h,w,c]=size(I)
    blockSigma = np.sqrt(blockVar)
    # Compute ratio of center and surround standard deviation
    cenSurDev = centerSurDev(Block, blockSize)
    # Relation between center-surround deviation and the block standard deviation
    blockBeta = (abs(blockSigma-cenSurDev))/(max(blockSigma, cenSurDev))

    return blockSigma, blockBeta

# Function to compute center surround Deviation of a block
def centerSurDev(Block, blockSize):
    # block center
    center1 = int((blockSize+1)/2)-1
    center2 = center1+1
    center = np.vstack((Block[:, center1], Block[:, center2]))
    # block surround
    Block = np.delete(Block, center1, axis=1)
    Block = np.delete(Block, center1, axis=1)

    # Compute standard deviation of block center and block surround
    center_std = np.std(center)
    surround_std = np.std(Block)

    # Ratio of center and surround standard deviation
    cenSurDev = (center_std/surround_std)

    # Check for nan's
    # if(isnan(cenSurDev)):
    #     cenSurDev = 0

    return cenSurDev
def calc_piqe(im):
    blockSize = 16  # Considered 16x16 block size for overall analysis
    activityThreshold = 0.1  # Threshold used to identify high spatially prominent blocks
    blockImpairedThreshold = 0.1  # Threshold identify blocks having noticeable artifacts
    windowSize = 6  # Considered segment size in a block edge.
    nSegments = blockSize-windowSize+1  # Number of segments for each block edge
    distBlockScores = 0  # Accumulation of distorted block scores
    NHSA = 0  # Number of high spatial active blocks.

    # pad if size is not divisible by blockSize
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    originalSize = im.shape
    rows, columns = originalSize
    rowsPad = rows % blockSize
    columnsPad = columns % blockSize
    isPadded = False
    if(rowsPad > 0 or columnsPad > 0):
        if rowsPad > 0:
            rowsPad = blockSize-rowsPad
        if columnsPad > 0:
            columnsPad = blockSize-columnsPad
        isPadded = True
        padSize = [rowsPad, columnsPad]
    im = np.pad(im, ((0, rowsPad), (0, columnsPad)), 'edge')

    # Normalize image to zero mean and ~unit std
    # used circularly-symmetric Gaussian weighting function sampled out
    # to 3 standard deviations.
    imnorm = calculate_mscn(im)

    # Preallocation for masks
    NoticeableArtifactsMask = np.zeros(imnorm.shape)
    NoiseMask = np.zeros(imnorm.shape)
    ActivityMask = np.zeros(imnorm.shape)

    # Start of block by block processing
    total_var = []
    total_bscore = []
    total_ndc = []
    total_nc = []

    BlockScores = []
    for i in np.arange(0, imnorm.shape[0]-1, blockSize):
        for j in np.arange(0, imnorm.shape[1]-1, blockSize):
             # Weights Initialization
            WNDC = 0
            WNC = 0

            # Compute block variance
            Block = imnorm[i:i+blockSize, j:j+blockSize]
            blockVar = np.var(Block)

            if(blockVar > activityThreshold):
                ActivityMask[i:i+blockSize, j:j+blockSize] = 1
                NHSA = NHSA+1

                # Analyze Block for noticeable artifacts
                blockImpaired = noticeDistCriterion(
                    Block, nSegments, blockSize-1, windowSize, blockImpairedThreshold, blockSize)

                if(blockImpaired):
                    WNDC = 1
                    NoticeableArtifactsMask[i:i +
                                            blockSize, j:j+blockSize] = blockVar

                # Analyze Block for guassian noise distortions
                [blockSigma, blockBeta] = noiseCriterion(
                    Block, blockSize-1, blockVar)

                if((blockSigma > 2*blockBeta)):
                    WNC = 1
                    NoiseMask[i:i+blockSize, j:j+blockSize] = blockVar

                # Pooling/ distortion assigment
                # distBlockScores = distBlockScores + \
                #     WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2)

                if WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2) > 0:
                    BlockScores.append(
                        WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2))

                total_var = [total_var, blockVar]
                total_bscore = [total_bscore, WNDC *
                                (1-blockVar) + WNC*(blockVar)]
                total_ndc = [total_ndc, WNDC]
                total_nc = [total_nc, WNC]

    BlockScores = sorted(BlockScores)
    lowSum = sum(BlockScores[:int(0.1*len(BlockScores))])
    Sum = sum(BlockScores)
    Scores = [(s*10*lowSum)/Sum for s in BlockScores]
    C = 1
    Score = ((sum(Scores) + C)/(C + NHSA))*100

    # if input image is padded then remove those portions from ActivityMask,
    # NoticeableArtifactsMask and NoiseMask and ensure that size of these masks
    # are always M-by-N.
    if(isPadded):
        NoticeableArtifactsMask = NoticeableArtifactsMask[0:originalSize[0],
                                                          0:originalSize[1]]
        NoiseMask = NoiseMask[0:originalSize[0], 0:originalSize[1]]
        ActivityMask = ActivityMask[0:originalSize[0], 1:originalSize[1]]

    return Score
