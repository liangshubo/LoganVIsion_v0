#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：seg_train_unet
@File    ：unet.py
@IDE     ：PyCharm
@Author  ：zxq
@Date    ：2021/12/16 15:35
"""

import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F

# 适用于 脊骨的切面约束和切面引导的，两个输入，一个是切面的数字和输入的cat 另一个是 channel_idx


# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out




class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.size = 32

        #  channel idx _ increase
        self.channel_insrease1 = nn.Sequential(
            nn.Conv2d(out_ch, 64, 1,groups=8),
            nn.BatchNorm2d(64))

        self.channel_insrease2 = nn.Sequential(
            nn.Conv2d(64, 256, 1, groups=8),
            nn.BatchNorm2d(256))

        self.channel_insrease3 = nn.Sequential(
            nn.Conv2d(256, 1024, 1, groups=8),
            nn.BatchNorm2d(1024))

        self.channel_insrease4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, groups=8),
            nn.BatchNorm2d(256))

        self.channel_insrease5 = nn.Sequential(
            nn.Conv2d(256, 64, 1, groups=8),
            nn.BatchNorm2d(64))




    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.size - h % self.size) % self.size
        mod_pad_w = (self.size - w % self.size) % self.size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, listinput):
        x = listinput[0]
        plane_channerl_idx = listinput[1]
        b, c =plane_channerl_idx.shape
        channel = plane_channerl_idx.view(b, c, 1, 1).to(torch.float)




        H, W = x.shape[2:]
        x = self.check_image_size(x)
        #h1, w1 = x.shape[2:]
        channel1= self.channel_insrease1(channel)  # 64
        channel2 = self.channel_insrease2(channel1)  #  256,
        channel3 = self.channel_insrease3(channel2)   # 1024
        channel4 = self.channel_insrease4(channel3)  #  256,
        channel5 = self.channel_insrease5(channel4)  # 64


        c1 = self.conv1(x)*channel1
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)*channel2
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)*channel3
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)*channel4
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)*channel5
        c10 = self.conv10(c9)
        out = F.interpolate(c10, (H, W))
      #  out = nn.Sigmoid()(out)

        out = out * channel
        return out
def make_model(args):
    return Unet(2,args.num_class)
if __name__ == '__main__':
    x = torch.randn([10,2,512,512]).cuda()
    channel = torch.randn([10,24]).cuda()
    model =  Unet(2,24).cuda()
    out = model([x,channel])
    print(out.shape)