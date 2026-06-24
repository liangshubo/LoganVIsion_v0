import json
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


"""
    逐层卷积
"""


class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=8):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        #打乱了通道的顺序
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w) #[ B，G，C//G , H,W]
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0) # C//2G ，1，1
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

class Res_SA_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_SA_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.attn = sa_layer(out_channels)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attn(out)

        out += residual
        out = self.relu(out)
        return out



class DepthwiseConv(nn.Module):
    """
        等价于一个分组数等于通道数的分组卷积
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小，元组类型
        padding: 补充
        stride: 步长
    """

    def __init__(self, in_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False):
        super(DepthwiseConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
            bias=bias
        )

    def forward(self, x):
        out = self.conv(x)
        return out


"""
    逐点卷积
"""


class PointwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


"""
    深度可分离卷积
"""


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        super(DepthwiseSeparableConv, self).__init__()

        self.conv1 = DepthwiseConv(
            in_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

        self.conv2 = PointwiseConv(
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


"""
    下采样
    [batch_size, in_channels, height, width] -> [batch_size, out_channels, height // stride, width // stride]
"""


class DownSampling(nn.Module):
    """
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        norm_layer: 正则化层，如果为None，使用BatchNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=None):
        super(DownSampling, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[-1] // 2)
        )

        if norm_layer is None:
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.norm = norm_layer

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return out

import json
import math

import torch.nn as nn
import torch

import torch.nn.functional as F

# from abc import *
# import utils

"""
    [batch_size, in_channels, height, width] -> [batch_size, out_channels, height // 4, width // 4]
"""


class StemConv(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            DownSampling(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
            DownSampling(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
        )

    def forward(self, x):
        out = self.proj(x)
        return out


class MSCA(nn.Module):

    def __init__(self, in_channels):
        super(MSCA, self).__init__()
        # 等价于521 的分组卷积 ，分组数等于 输入通道数
        self.conv = DepthwiseConv(
            in_channels=in_channels,
            kernel_size=(5, 5),
            padding=(2, 2),
            bias=True
        )

        self.conv7 = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                bias=True
            ),
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                bias=True
            )
        )

        self.conv11 = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 11),
                padding=(0, 5),
                bias=True
            ),
           DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(11, 1),
                padding=(5, 0),
                bias=True
            )
        )

        self.conv21 = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 21),
                padding=(0, 10),
                bias=True
            ),
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(21, 1),
                padding=(10, 0),
                bias=True
            )
        )
        self.sa_layer =  sa_layer(in_channels)


        self.fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        u = x
        out = self.conv(x)

        branch1 = self.conv7(out)
        branch2 = self.conv11(out)
        branch3 = self.conv21(out)

        out = self.fc(out + branch1 + branch2 + branch3)
        out = out * u
        out = self.sa_layer(out)
        return out

class MSCA_Large(nn.Module):
    '''
    增大的 卷积核  同时 注意力的 模块更改为残差块     DepthwiseSeparableConv
    '''
    def __init__(self, in_channels):
        super(MSCA_Large, self).__init__()
        # 等价于521 的分组卷积 ，分组数等于 输入通道数
        self.conv = DepthwiseSeparableConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(7, 7),
            padding=(3, 3)
        )

        self.conv11 = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 11),
                padding=(0, 5)
            ),
            DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(11, 1),
                padding=(5, 0)
            )
        )

        self.conv15 = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 15),
                padding=(0, 7)
            ),
           DepthwiseSeparableConv(
                in_channels=in_channels,
               out_channels=in_channels,
                kernel_size=(15, 1),
                padding=(7, 0)
            )
        )

        self.conv21 = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 21),
                padding=(0, 10)
            ),
            DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(21, 1),
                padding=(10, 0)
            )
        )
        self.sa_layer =  Res_SA_block(in_channels,in_channels)


        self.fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        u = x
        out = self.conv(x)

        branch1 = self.conv11(out)
        branch2 = self.conv15(out)
        branch3 = self.conv21(out)
       #  print(branch1.shape ,branch2.shape , branch3.shape)
        out = self.fc(out + branch1 + branch2 + branch3)
        out = out * u
        out = self.sa_layer(out)
        return out


class Attention(nn.Module):

    def __init__(self, in_channels):
        super(Attention, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )
        self.msca = MSCA(in_channels=in_channels)
        self.fc2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        out = F.gelu(self.fc1(x))
        out = self.msca(out)
        out = self.fc2(out)
        return out
class Attention_Large(nn.Module):

    def __init__(self, in_channels):
        super(Attention_Large, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )
        self.msca = MSCA_Large(in_channels=in_channels)
        self.fc2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        out = F.gelu(self.fc1(x))
        out = self.msca(out)
        out = self.fc2(out)
        return out


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, drop_prob=0.):
        super(FFN, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=(1, 1)
        )
        self.dw = DepthwiseConv(
            in_channels=hidden_features,
            kernel_size=(3, 3),
            bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=(1, 1)
        )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        out = self.fc1(x)
        out = F.gelu(self.dw(out))
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):

    def __init__(self, in_channels, expand_ratio, drop_prob=0., drop_path_prob=0.):
        super(Block, self).__init__()

        self.norm1 = nn.BatchNorm2d(num_features=in_channels)
        self.attention = Attention(in_channels=in_channels)
        self.drop_path = DropPath(drop_prob=drop_path_prob if drop_path_prob >= 0 else nn.Identity)
        self.norm2 = nn.BatchNorm2d(num_features=in_channels)
        self.ffn = FFN(
            in_features=in_channels,
            hidden_features=int(expand_ratio * in_channels),
            out_features=in_channels,
            drop_prob=drop_prob
        )

        layer_scale_init_value = 1e-2
        self.layer_scale1 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )
        self.layer_scale2 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )

    def forward(self, x):
        out = self.norm1(x)
        out = self.attention(out)
        out = x + self.drop_path(
            self.layer_scale1.unsqueeze(-1).unsqueeze(-1) * out
        )
        x = out

        out = self.norm2(out)
        out = self.ffn(out)
        out = x + self.drop_path(
            self.layer_scale2.unsqueeze(-1).unsqueeze(-1) * out
        )

        return out

class Block_Large(nn.Module):

    def __init__(self, in_channels, expand_ratio, drop_prob=0., drop_path_prob=0.):
        super(Block_Large, self).__init__()

        self.norm1 = nn.BatchNorm2d(num_features=in_channels)
        self.attention = Attention_Large(in_channels=in_channels)
        self.drop_path = DropPath(drop_prob=drop_path_prob if drop_path_prob >= 0 else nn.Identity)
        self.norm2 = nn.BatchNorm2d(num_features=in_channels)
        self.ffn = FFN(
            in_features=in_channels,
            hidden_features=int(expand_ratio * in_channels),
            out_features=in_channels,
            drop_prob=drop_prob
        )

        layer_scale_init_value = 1e-2
        self.layer_scale1 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )
        self.layer_scale2 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )

    def forward(self, x):
        out = self.norm1(x)
        out = self.attention(out)
        out = x + self.drop_path(
            self.layer_scale1.unsqueeze(-1).unsqueeze(-1) * out
        )
        x = out

        out = self.norm2(out)
        out = self.ffn(out)
        out = x + self.drop_path(
            self.layer_scale2.unsqueeze(-1).unsqueeze(-1) * out
        )

        return out


class Stage(nn.Module):

    def __init__(
            self,
            stage_id,
            in_channels,
            out_channels,
            expand_ratio,
            blocks_num,
            drop_prob=0.,
            drop_path_prob=[0.]
    ):
        super(Stage, self).__init__()

        assert blocks_num == len(drop_path_prob)

        if stage_id == 0:
            self.down_sampling = StemConv(
                in_channels=in_channels,
                out_channels=out_channels
            )
        else:
            self.down_sampling = DownSampling(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2)
            )

        self.blocks = nn.Sequential(
            *[
                Block(
                    in_channels=out_channels,
                    expand_ratio=expand_ratio,
                    drop_prob=drop_prob,
                    drop_path_prob=drop_path_prob[i]
                ) for i in range(0, blocks_num)
            ]
        )

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        out = self.down_sampling(x)
        out = self.blocks(out)
        # [batch_size, channels, height, width] -> [batch_size, channels, height * width]
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1)
        # [batch_size, channels, height * width] -> [batch_size, height * width, channels]
        out = torch.transpose(out, -2, -1)
        out = self.norm(out)

        # [batch_size, height * width, channels] -> [batch_size, channels, height * width]
        out = torch.transpose(out, -2, -1)
        # [batch_size, channels, height * width] -> [batch_size, channels, height, width]
        out = out.view(batch_size, -1, height, width)

        return out
class Stage_Large(nn.Module):

    def __init__(
            self,
            stage_id,
            in_channels,
            out_channels,
            expand_ratio,
            blocks_num,
            drop_prob=0.,
            drop_path_prob=[0.]
    ):
        super(Stage_Large, self).__init__()

        assert blocks_num == len(drop_path_prob)

        if stage_id == 0:
            self.down_sampling = StemConv(
                in_channels=in_channels,
                out_channels=out_channels
            )
        else:
            self.down_sampling = DownSampling(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2)
            )

        self.blocks = nn.Sequential(
            *[
                Block_Large(
                    in_channels=out_channels,
                    expand_ratio=expand_ratio,
                    drop_prob=drop_prob,
                    drop_path_prob=drop_path_prob[i]
                ) for i in range(0, blocks_num)
            ]
        )

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        out = self.down_sampling(x)
        out = self.blocks(out)
        # [batch_size, channels, height, width] -> [batch_size, channels, height * width]
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1)
        # [batch_size, channels, height * width] -> [batch_size, height * width, channels]
        out = torch.transpose(out, -2, -1)
        out = self.norm(out)

        # [batch_size, height * width, channels] -> [batch_size, channels, height * width]
        out = torch.transpose(out, -2, -1)
        # [batch_size, channels, height * width] -> [batch_size, channels, height, width]
        out = out.view(batch_size, -1, height, width)

        return out


class MSCAN(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 64, 160, 256],  # 这里将会
            expand_ratios=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob=0.1,
            drop_path_prob=0.1
    ):
        super(MSCAN, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, sum(depths))]  # 往后drop 的概率就越大
        self.stages = nn.Sequential(
            *[
                Stage(
                    stage_id=stage_id,
                    in_channels=embed_dims[stage_id],
                    out_channels=embed_dims[stage_id + 1],
                    expand_ratio=expand_ratios[stage_id],  # 块内的 通道扩展比例
                    blocks_num=depths[stage_id],  # 此stage  有多少的 块的数量
                    drop_prob=drop_prob,  # 神经元有多少的概率其丢弃
                    drop_path_prob=dpr[sum(depths[: stage_id]): sum(depths[: stage_id + 1])]  # 块的范围内的下的drop的数量
                ) for stage_id in range(0, len(depths))  # stage = 0,1,2,3
            ]
        )

    def forward(self, x):
        out = x
        outputs = []

        for idx, stage in enumerate(self.stages):
            out = stage(out)
            if idx != 0:
                outputs.append(out)

        # outputs: [output_of_stage1, output_of_stage2, output_of_stage3]
        # output_of_stage1: [batch_size, embed_dims[2], height / 8, width / 8]
        # output_of_stage2: [batch_size, embed_dims[3], height / 16, width / 16]
        # output_of_stage3: [batch_size, embed_dims[4], height / 32, width / 32]
        return [x, *outputs]

class MSCAN_Large(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 64, 160, 256],  # 这里将会
            expand_ratios=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob=0.1,
            drop_path_prob=0.1
    ):
        super(MSCAN_Large, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, sum(depths))]  # 往后drop 的概率就越大
        self.stages = nn.Sequential(
            *[
                Stage_Large(
                    stage_id=stage_id,
                    in_channels=embed_dims[stage_id],
                    out_channels=embed_dims[stage_id + 1],
                    expand_ratio=expand_ratios[stage_id],  # 块内的 通道扩展比例
                    blocks_num=depths[stage_id],  # 此stage  有多少的 块的数量
                    drop_prob=drop_prob,  # 神经元有多少的概率其丢弃
                    drop_path_prob=dpr[sum(depths[: stage_id]): sum(depths[: stage_id + 1])]  # 块的范围内的下的drop的数量
                ) for stage_id in range(0, len(depths))  # stage = 0,1,2,3
            ]
        )

    def forward(self, x):
        out = x
        outputs = []

        for idx, stage in enumerate(self.stages):
            out = stage(out)
            if idx != 0:
                outputs.append(out)

        # outputs: [output_of_stage1, output_of_stage2, output_of_stage3]
        # output_of_stage1: [batch_size, embed_dims[2], height / 8, width / 8]
        # output_of_stage2: [batch_size, embed_dims[3], height / 16, width / 16]
        # output_of_stage3: [batch_size, embed_dims[4], height / 32, width / 32]
        return [x, *outputs]


class LightHamHead_Msk(nn.Module):
    # 为了进行脊骨 分割 ，需要引入额外的组织权重
    def __init__(
            self,
            in_channels_list=[64, 160, 256],
            hidden_channels=256,
            out_channels=256,
            classes_num=150,
            drop_prob=0.1
    ):
        super(LightHamHead_Msk, self).__init__()

        self.cls_seg = nn.Sequential(
            nn.Dropout2d(drop_prob),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=classes_num,
                kernel_size=(1, 1)
            )
        )

        self.squeeze = nn.Sequential(
            nn.Conv2d(
                in_channels=sum(in_channels_list),
                out_channels=hidden_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hidden_channels,
            ),
            nn.GELU()
        )



        self.align = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=out_channels
            ),
            nn.GELU()
        )


    # inputs: [x, x_1, x_2, x_3]
    # x: [batch_size, channels, height, width]
    def forward(self, inputs):

        # 引入 切面的组织分割权重

        assert len(inputs) >= 2
        o = inputs[0]
        batch_size, _, standard_height, standard_width = inputs[1].shape
        standard_shape = (standard_height, standard_width)
        inputs = [
            F.interpolate(
                input=x,
                size=standard_shape,
                mode="bilinear",
                align_corners=False
            )
            for x in inputs[1:]
        ]

        # x: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        x = torch.cat(inputs, dim=1)

        # out: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        out = self.squeeze(x)

        # 切面 -------


        # ---------
        # out = self.hamburger(out)
        out = self.align(out)

        # out: [batch_size, classes_num, standard_height, standard_width]
        out = self.cls_seg(out)



        _, _, original_height, original_width = o.shape
        # out: [batch_size, original_height * original_width, classes_num]
        out = F.interpolate(
            input=out,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )
        #print(out.shape)

        #out = torch.transpose(out.view(batch_size, -1, original_height * original_width), -2, -1)
        #print(out.shape)
        return out

class SegNeXt_MSK(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 64, 160, 256],  # 维度 信息 各个stage
            expand_rations=[8, 8, 4, 4],  # 通道的比例
            depths=[3, 3, 5, 2],  # 块的数量
            drop_prob_of_encoder=0.1,  # 编码器的drop
            drop_path_prob=0.1,  #
            hidden_channels=256,  # 隐藏层 通道 ？
            out_channels=256,  # 输出通道？
            classes_num=150,  # 类别数
            drop_prob_of_decoder=0.1,  # 解码器的drop

    ):
        super( SegNeXt_MSK, self).__init__()

        self.encoder = MSCAN(
            embed_dims=embed_dims,
            expand_ratios=expand_rations,
            depths=depths,
            drop_prob=drop_prob_of_encoder,
            drop_path_prob=drop_path_prob
        )

        self.decoder = LightHamHead_Msk(
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            classes_num=classes_num,
            drop_prob=drop_prob_of_decoder
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x ):

        out = self.encoder(x)
        out = self.decoder(out)
        return out


class SegNeXt_MSK_Large(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 64, 160, 256],  # 维度 信息 各个stage
            expand_rations=[8, 8, 4, 4],  # 通道的比例
            depths=[3, 3, 5, 2],  # 块的数量
            drop_prob_of_encoder=0.1,  # 编码器的drop
            drop_path_prob=0.1,  #
            hidden_channels=256,  # 隐藏层 通道 ？
            out_channels=256,  # 输出通道？
            classes_num=150,  # 类别数
            drop_prob_of_decoder=0.1,  # 解码器的drop

    ):
        super( SegNeXt_MSK_Large, self).__init__()

        self.encoder = MSCAN_Large(
            embed_dims=embed_dims,
            expand_ratios=expand_rations,
            depths=depths,
            drop_prob=drop_prob_of_encoder,
            drop_path_prob=drop_path_prob
        )

        self.decoder = LightHamHead_Msk(
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            classes_num=classes_num,
            drop_prob=drop_prob_of_decoder
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x ):

        out = self.encoder(x)
        out = self.decoder(out)
        return out


def make_model(args):
    return   SegNeXt_MSK_Sip(args)


def SegNeXt_MSK_Sip(args):
    return   SegNeXt_MSK(embed_dims=[1, 64, 128, 320, 512],  # 维度 信息 各个stage
                    expand_rations=[8, 8, 4, 4],  # 通道的比例
                    depths=[4, 4, 16, 4],  # 块的数量
                    drop_prob_of_encoder=0,  # 编码器的drop
                    drop_path_prob=0,  #
                    hidden_channels=256,  # 隐藏层 通道 ？
                    out_channels=256,  # 输出通道？
                    classes_num=args.num_class,  # 类别数
                    drop_prob_of_decoder=0.1)   #


def SegNeXt_MSK_Mid(args):
    return   SegNeXt_MSK(embed_dims=[1, 64, 128, 320, 512],  # 维度 信息 各个stage
                    expand_rations=[8, 8, 8, 4],  # 通道的比例
                    depths=[4, 4, 16, 4],  # 块的数量
                    drop_prob_of_encoder=0,  # 编码器的drop
                    drop_path_prob=0,  #
                    hidden_channels=256,  # 隐藏层 通道 ？
                    out_channels=256,  # 输出通道？
                    classes_num=args.num_class,  # 类别数
                    drop_prob_of_decoder=0.1)   #


def SegNeXt_MSK_Lar(args):
    return   SegNeXt_MSK_Large(embed_dims=[1, 64, 128, 320, 512],  # 维度 信息 各个stage
                    expand_rations=[4, 6, 8, 4],  # 通道的比例
                    depths=[3, 4, 12, 4],  # 块的数量
                    drop_prob_of_encoder=0,  # 编码器的drop
                    drop_path_prob=0,  #
                    hidden_channels=256,  # 隐藏层 通道 ？
                    out_channels=256,  # 输出通道？
                    classes_num=args.num_class,  # 类别数
                    drop_prob_of_decoder=0.1)   #



if __name__ == '__main__':

    class arg:
        def __init__(self):
            self.num_class = 2

    args = arg()

    x = torch.rand([1, 1, 256, 256]).cuda()
    model = SegNeXt_MSK_Lar(args)
    model = model.cuda()
    print(model)
    print(model(x).shape)
    print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (
            sum(p.numel() for p in model.parameters()) / 1000000.0))


