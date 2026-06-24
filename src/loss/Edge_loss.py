import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from  scipy.ndimage import  distance_transform_edt

import cv2

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class edge_loss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self):
        super(edge_loss, self).__init__()
        self.sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.mse  = nn.MSELoss()
    def forward(self,output,label):
        edge_output = self.sobel_operator(output)
        edge_label = self.sobel_operator(label)
        edge_loss = self.mse(edge_output,edge_label)
        if edge_loss > 1000:
            edge_loss = edge_loss / 1000
        return edge_loss
        
    def sobel_operator(self,img_tensor):
        # 将图像转换为PyTorch的Tensor
        # 定义Sobel滤波器
        # 对图像进行卷积操作
        grad_x = F.conv2d(img_tensor, self.sobel_x.cuda().unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = F.conv2d(img_tensor, self.sobel_y.cuda().unsqueeze(0).unsqueeze(0), padding=1)

        # 计算梯度幅值
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        # 转换为NumPy数组
        return grad_magnitude


class BoundaryLoss(nn.Module):
    def __init__(self, theta=10.0):
        super().__init__()
        self.theta = theta  # 控制边界敏感度的带宽参数
        # 适用于多类别的语义分割框架 ，其中标签为P模式的B，H，W 类型为long 其中数值0-numclass ,预测为B，C，H，W ，类型为float,C=num_class ;
    def forward(self, pred, target):
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # 生成边界距离变换图
        with torch.no_grad():
            boundary_dist = self._compute_distance_transform(target_onehot)

        pred_softmax = F.softmax(pred, dim=1)
        loss = torch.mean(boundary_dist * pred_softmax)
        return loss

    def _compute_distance_transform(self, target):
        # 使用scipy.ndimage.distance_transform_edt计算（需转为CPU）
        dist_maps = []
        for b in range(target.shape[0]):
            for c in range(target.shape[1]):
                binary_img = target[b, c].cpu().numpy()
                dist_map = distance_transform_edt(binary_img)  # 欧氏距离变换
                dist_maps.append(torch.from_numpy(dist_map))
        return torch.stack(dist_maps).to(target.device)