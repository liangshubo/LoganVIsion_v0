import torch
import torch.nn as nn
import torch.nn.functional as F
def SoftIoULossf(pred, target):
    # Old One
    pred = torch.sigmoid(pred)  # 经过一个sigmoid层
    smooth = 1

    # print("pred.shape: ", pred.shape)
    # print("target.shape: ", target.shape)

    intersection = pred * target  # 这是什么意思，主要是分割中的软iou损失  ，
    # 确实可以这是一个与运算，只有不等于的0的才可以为1，也就是，不管预测的是什么，当且仅当pred预测不是0，而且标签的位置也不是0才会不是0
    # 因此可以直接相乘，得到两者的交集

    # 交集不应该是 pred*(pre == target).float ? ??
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)
    # 这就是计算iou

    # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
    #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
    #         - intersection.sum(axis=(1, 2, 3)) + smooth)

    loss = 1 - loss.mean()
    # loss = (1 - loss).mean()
    # 想要iou足够大 ，那么就将1-iou足够小就可以
    return loss
class SoftIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,pred,label):
        loss  = self.SoftIoULoss(pred,label)
        return loss

    def SoftIoULoss(self,pred, target):
        # Old One
        pred = torch.sigmoid(pred)  # 经过一个sigmoid层
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target  # 这是什么意思，主要是分割中的软iou损失  ，
        # 确实可以这是一个与运算，只有不等于的0的才可以为1，也就是，不管预测的是什么，当且仅当pred预测不是0，而且标签的位置也不是0才会不是0
        # 因此可以直接相乘，得到两者的交集

        # 交集不应该是 pred*(pre == target).float ? ??
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)
        # 这就是计算iou

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()
        # 想要iou足够大 ，那么就将1-iou足够小就可以
        return loss


class IoULoss(nn.Module):
    # 适用于多类别的语义分割框架 ，其中标签为P模式的B，H，W 类型为long 其中数值0-numclass ,预测为B，C，H，W ，类型为float,C=num_class ;
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        pred_softmax = F.softmax(pred, dim=1)

        intersection = torch.sum(pred_softmax * target_onehot, dim=(2, 3))
        total = torch.sum(pred_softmax, dim=(2, 3)) + torch.sum(target_onehot, dim=(2, 3))
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()