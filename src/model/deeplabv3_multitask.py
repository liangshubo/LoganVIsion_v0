import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import torchvision


# 整合了 resnet 的 deeplab v3   有两个输出头

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.transforms.v2.functional import elastic

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



# Attention
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  # width = 64,g32 . w = 4    64 * 4/64 * 32  = 128
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# attention

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


# Attention
class AttentionBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(AttentionBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  # width = 64,g32 . w = 4    64 * 4/64 * 32  = 128
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Sequential(conv3x3(width, width, stride, groups, dilation),CBAM(width,width,ratio=16))
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.selu = nn.SELU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.selu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.selu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.selu(out)

        return out


class ResNetDeepLab(nn.Module):

    def __init__(self, block, layers, plane_num_classes=11, zero_init_residual=False,
                 groups=1, width_per_group=64, segment_num_classes=24, output_stride=8,
                 norm_layer=None):

        # group 32  /  wid = 4
        super(ResNetDeepLab, self).__init__()

        # ----------------------以下是 deeplabv3plus的初始化 --------------------
        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]  # 空洞卷积隔的步数

        inplanes = 2048
        low_level_planes = 256

        self.project = nn.Sequential(
            nn.Conv2d(low_level_planes, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP( inplanes, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, segment_num_classes, 1)
        )


        # ------------------------以下是 resnet 的 初始化   ------------------
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #     INPUT   CHANNEL
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])    # output_stride = [false true true ]  dilation = 1
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])    # dilation = 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])     # dilation =  4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc =  nn.Sequential(nn.Linear(512 * block.expansion,plane_num_classes),nn.Sigmoid())

        for m in self.modules():  # 初始化权重参数
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,  AttentionBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))   # 第一个块 负责降采样， 空洞率是后面的一半
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        input_shape = x.shape[-2:]

        x = self.conv1(x)   # h /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # h /4   channel = pplane  * exp
        layer1 = self.layer1(x)   # h /4   channel = pplane  * exp

        #print(layer1.shape,"layer1.shape")
        layer2 = self.layer2(layer1)    # h / 8   channel = pplane  * exp
        #print(layer2.shape, "layer2.shape")
        layer3 = self.layer3(layer2)     # h / 16.  channel = pplane  * exp
        #print(layer3.shape, "layer3.shape")
        layer4 = self.layer4(layer3)     # h / 32   channel = pplane  * exp
        #print(layer4.shape, "layer4.shape")
        avgpool = self.avgpool(layer4)
        fla = torch.flatten(avgpool, 1)
        class_output = self.fc(fla)
        #print(class_output.shape, "class_output.shape")
        # ----------------------- 上面是切面分类的结果 / 下面是分割的结果    ----------------------------

        low_level_feature = self.project(layer1)  # 1*1的卷积+bn+relu
        #print(low_level_feature.shape, "low_level_feature.shape")

        output_feature = self.aspp(layer4)
        #print(output_feature.shape, "output_feature.shape")

        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:],
                                       mode='bicubic', align_corners=False)  # 做个上采样
        #print(output_feature.shape, "output_feature.shape")

        init_segment_output= self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

        segment_output = F.interpolate(init_segment_output, size=input_shape, mode='bicubic', align_corners=False)
        #print(segment_output.shape, "segment_output.shape")

        return class_output ,  segment_output


class ResNetDeepLab_plus(nn.Module):

    def __init__(self, block, layers, plane_num_classes=11, zero_init_residual=False,
                 groups=1, width_per_group=64, segment_num_classes=24, output_stride=8,
                 norm_layer=None):

        # group 32  /  wid = 4
        super(ResNetDeepLab_plus, self).__init__()

        # ----------------------以下是 deeplabv3plus的初始化 --------------------
        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]  # 空洞卷积隔的步数

        inplanes = 2048
        low_level_planes = 256

        self.project = nn.Sequential(
            nn.Conv2d(low_level_planes, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP( inplanes, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, segment_num_classes, 1)
        )

        # 分类引导分割


        # ------------------------以下是 resnet 的 初始化   ------------------
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #     INPUT   CHANNEL
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])    # output_stride = [false true true ]  dilation = 1
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])    # dilation = 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])     # dilation =  4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc =  nn.Sequential(nn.Linear(512 * block.expansion,plane_num_classes),nn.Sigmoid())

        for m in self.modules():  # 初始化权重参数
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,  AttentionBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.initclass_lead_segment = nn.Sequential(
            nn.Conv2d(plane_num_classes,plane_num_classes*4,1),
            nn.BatchNorm2d(plane_num_classes*4),
            nn.GELU(),
            nn.Conv2d(plane_num_classes * 4,128, 1,groups=4),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128,64, 1,groups=4),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, segment_num_classes, 1, groups=4),
            nn.BatchNorm2d(segment_num_classes)
        )
        self.initsegment_lead_class1 = nn.Sequential(
            nn.Conv2d(segment_num_classes, segment_num_classes * 4, 3,1,1),
            nn.BatchNorm2d(segment_num_classes * 4),
            nn.GELU(),
            nn.Conv2d(segment_num_classes * 4, 128, 5,2,1 ,groups=4),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveMaxPool2d((1, 1)),

        )
        self.initsegment_lead_class2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, plane_num_classes),
        )






    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))   # 第一个块 负责降采样， 空洞率是后面的一半
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        input_shape = x.shape[-2:]

        x = self.conv1(x)   # h /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # h /4   channel = pplane  * exp
        layer1 = self.layer1(x)   # h /4   channel = pplane  * exp

        #print(layer1.shape,"layer1.shape")
        layer2 = self.layer2(layer1)    # h / 8   channel = pplane  * exp
        #print(layer2.shape, "layer2.shape")
        layer3 = self.layer3(layer2)     # h / 16.  channel = pplane  * exp
        #print(layer3.shape, "layer3.shape")
        layer4 = self.layer4(layer3)     # h / 32   channel = pplane  * exp
        #print(layer4.shape, "layer4.shape")
        avgpool = self.avgpool(layer4)
        fla = torch.flatten(avgpool, 1)
        init_class_output = self.fc(fla)

        class_lead_segment = init_class_output.unsqueeze(2).unsqueeze(3)

        #print(init_class_output.shape, "init_class_output.shape")
        #print( class_lead_segment.shape, " class_lead_segment")

        class_lead_segment = self.initclass_lead_segment(class_lead_segment)
        # ----------------------- 上面是切面分类的结果 / 下面是分割的结果    ----------------------------

        low_level_feature = self.project(layer1)  # 1*1的卷积+bn+relu
        #print(low_level_feature.shape, "low_level_feature.shape")

        output_feature = self.aspp(layer4)
        #print(output_feature.shape, "output_feature.shape")

        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:],
                                       mode='bicubic', align_corners=False)  # 做个上采样
        #print(output_feature.shape, "output_feature.shape")

        init_segment_output = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

        #print(init_segment_output.shape )
        segment_lead_class = self.initsegment_lead_class1( init_segment_output)
        #print(segment_lead_class.shape," segment_lead_class ")
        segment_lead_class = torch.flatten(segment_lead_class, 1)
        segment_lead_class2 = self.initsegment_lead_class2(segment_lead_class)
        segment_output = init_segment_output * class_lead_segment

        segment_output = F.interpolate(segment_output, size=input_shape, mode='bicubic', align_corners=False)

        init_segment_output = F.interpolate(init_segment_output, size=input_shape, mode='bicubic', align_corners=False)
        #print(segment_output.shape, "segment_output.shape")
        class_output = init_class_output * segment_lead_class2
        return init_class_output, class_output , init_segment_output, segment_output

class ResNetDeepLab_plusv2(nn.Module):

    def __init__(self, block, layers, plane_num_classes=11, zero_init_residual=False,
                 groups=1, width_per_group=64, segment_num_classes=24, output_stride=8,
                 norm_layer=None):

        # group 32  /  wid = 4
        super(ResNetDeepLab_plusv2, self).__init__()

        # ----------------------以下是 deeplabv3plus的初始化 --------------------
        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]  # 空洞卷积隔的步数

        inplanes = 2048
        low_level_planes = 256

        self.project = nn.Sequential(
            nn.Conv2d(low_level_planes, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP2( inplanes, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, segment_num_classes, 1)
        )

        # 分类引导分割


        # ------------------------以下是 resnet 的 初始化   ------------------
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #     INPUT   CHANNEL
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])    # output_stride = [false true true ]  dilation = 1
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])    # dilation = 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])     # dilation =  4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc =  nn.Sequential(nn.Linear(512 * block.expansion,plane_num_classes),nn.Sigmoid())

        for m in self.modules():  # 初始化权重参数
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,  AttentionBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


        self.initsegment_lead_class1 = nn.Sequential(
            nn.Conv2d(segment_num_classes, segment_num_classes * 4, 3,2,1),
            nn.BatchNorm2d(segment_num_classes * 4),
            nn.GELU(),
            nn.Conv2d(segment_num_classes * 4, 128, 5,2,1 ,groups=4),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveMaxPool2d((1, 1)),

        )
        self.initsegment_lead_class2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, plane_num_classes),
        )

        self.plane_channel_increse = nn.Sequential(
            nn.Conv2d(segment_num_classes, 256, 1,groups=8),
            nn.GELU(),
            nn.Conv2d(256, 1024, 1,groups=256),
            nn.GELU(),
            nn.Conv2d(1024, inplanes, 1,groups=256)
        )

        self.segment_num_classes = segment_num_classes



    def create_onehot_from_indices(self,plane_out):
        """
        将组织索引列表转换为固定维度的one-hot编码
        参数:
       plane_out
        dim -- one-hot编码的维度，默认为24
        返回:
        一个dim维的numpy数组，其中指定索引位置为1，其余为0
        """
        #  -------------------预定义的 -------------------------
        def _single_onehot_from_indices(plane):
            section_to_tissue = [  [0,0,0],[0,1, 2, 3],     # 切面1的组织索引
                                        [0,3, 4, 5],     # 切面2的组织索引
                                        [0,3, 6, 7],     # 切面3的组织索引
                                        [0,3, 8, 9],     # 切面4的组织索引
                                        [0,3, 10,11,12], # 切面5的组织索引
                                        [0,3, 13, 14],   # 切面6的组织索引
                                        [0,15, 16],      # 切面7的组织索引
                                        [0,3, 16, 17],   # 切面8的组织索引
                                        [0,18, 19, 20],  # 切面10的组织索引
                                        [0,21, 22, 23],  # 切面11的组织索引
                                     ]
            # 初始化全0向量
            indices_list = section_to_tissue[plane]
            onehot = torch.zeros(self.segment_num_classes, dtype=float).cuda()

            # 将指定索引位置设为1
            for idx in indices_list:  #这是切面的索引 对应的组织
                if 1 <= idx <=  self.segment_num_classes: # 这个是排除0
                    onehot[idx] = 1  # 索引从1开始，但数组从0开始

            return onehot.unsqueeze(0)

        b  = plane_out.shape[0]
        segment_channel_list = []
        for i in range(b):
            single_plane_out = plane_out[i]
            class_idx = torch.argmax(single_plane_out)
            segment_channel_list.append(_single_onehot_from_indices(class_idx))
        segment_channel = torch.cat(segment_channel_list,dim=0).unsqueeze(2).unsqueeze(3)
        return segment_channel


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))   # 第一个块 负责降采样， 空洞率是后面的一半
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        input_shape = x.shape[-2:]

        x = self.conv1(x)   # h /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # h /4   channel = pplane  * exp
        layer1 = self.layer1(x)   # h /4   channel = pplane  * exp
        #print(layer1.shape,"layer1.shape")
        layer2 = self.layer2(layer1)    # h / 8   channel = pplane  * exp
        #print(layer2.shape, "layer2.shape")
        layer3 = self.layer3(layer2)     # h / 16.  channel = pplane  * exp
        #print(layer3.shape, "layer3.shape")
        layer4 = self.layer4(layer3)     # h / 32   channel = pplane  * exp
        #print(layer4.shape, "layer4.shape")
        avgpool = self.avgpool(layer4)
        fla = torch.flatten(avgpool, 1)
        init_class_output = self.fc(fla)

        #-------------------------------- class lead segment  ---------------------

        class_lead_segment = self.create_onehot_from_indices(init_class_output ).to(torch.float)

        class_lead_aspp=self.plane_channel_increse(class_lead_segment)

        # ----------------------- 上面是切面分类的结果 / 下面是分割的结果    ----------------------------

        low_level_feature = self.project(layer1)  # 1*1的卷积+bn+relu
        #print(low_level_feature.shape, "low_level_feature.shape")

        output_feature = self.aspp(layer4*class_lead_aspp)
        #print(output_feature.shape, "output_feature.shape")

        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:],
                                       mode='bicubic', align_corners=False)  # 做个上采样

        init_segment_output = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

        segment_output = init_segment_output * class_lead_segment

        # segment lead class

        segment_lead_class = self.initsegment_lead_class1(segment_output)

        segment_lead_class = torch.flatten(segment_lead_class, 1)
        segment_lead_class2 = self.initsegment_lead_class2(segment_lead_class)

        segment_output = F.interpolate(segment_output, size=input_shape, mode='bicubic', align_corners=False)
        init_segment_output = F.interpolate(init_segment_output, size=input_shape, mode='bicubic', align_corners=False)
        #print(segment_output.shape, "segment_output.shape")
        class_output = init_class_output * segment_lead_class2

        return init_class_output, class_output , init_segment_output, segment_output


# -----------------------------------ASPP 相关 ----------------------------
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [  # 这个卷积就比正常卷积多了个dilation    padding和dilation一样为了确保输出特征图大小固定
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),   # 在K=3 的时候 Padding = Dilation  可以维持 输出的   特征图
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):   # input  2048   ,atrous_rate = [12, 24, 36]
        super(ASPP, self).__init__()
        out_channels = 256

        modules = [nn.Sequential(  # 第一个 先来个1*1的卷积+bn+relu
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))]

        rate1, rate2, rate3 = tuple(atrous_rates)    # 12 -24 --36
        modules.append(ASPPConv(in_channels, out_channels, rate1))  # 第二个 3*3的rate为6的空洞卷积
        modules.append(ASPPConv(in_channels, out_channels, rate2))  # 第三个 3*3的rate为12的空洞卷积
        modules.append(ASPPConv(in_channels, out_channels, rate3))  # 第四个 3*3的rate为18的空洞卷积
        modules.append(ASPPPooling(in_channels, out_channels))  # 第五个 pooling[1,1]+1*1卷积+bn+relu+resize

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )  # 这里居然有丢弃层 知道这个东西 从来没见过有人用

    def forward(self, x):
        res = []
        for conv in self.convs:
            # print(conv(x).shape)
            res.append(conv(x))
        res = torch.cat(res, dim=1)  # 将五个输出拼接在一起
        return self.project(res)  # 然后再做一个1*1卷积+bn+relu+丢弃层

class ASPP2(nn.Module):  # 250924   对分割模块进行增加
    def __init__(self, in_channels, atrous_rates):   # input  2048   ,atrous_rate = [12, 24, 36]
        super(ASPP2, self).__init__()
        out_channels = 256

        modules = [nn.Sequential(  # 第一个 先来个1*1的卷积+bn+relu
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))]

        rate1, rate2, rate3 = tuple(atrous_rates)    # 12 -24 --36
        modules.append(ASPPConv(in_channels, out_channels, rate1))  # 第二个 3*3的rate为6的空洞卷积
        modules.append(ASPPConv(in_channels, out_channels, rate2))  # 第三个 3*3的rate为12的空洞卷积
        modules.append(ASPPConv(in_channels, out_channels, rate3))  # 第四个 3*3的rate为18的空洞卷积
        modules.append(ASPPPooling(in_channels, out_channels))  # 第五个 pooling[1,1]+1*1卷积+bn+relu+resize

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )  # 这里居然有丢弃层 知道这个东西 从来没见过有人用

    def forward(self, x):
        res = []
        for conv in self.convs:
            # print(conv(x).shape)
            res.append(conv(x))
        res = torch.cat(res, dim=1)  # 将五个输出拼接在一起
        return self.project(res)  # 然后再做一个1*1卷积+bn+relu+丢弃层




#--------------------- Deeplab v3+--------------------

def deeplabv3plus_resnet50(block=AttentionBottleneck,  plane_num_classes=10,
                 segment_num_classes=24, output_stride=8):
    return ResNetDeepLab_plus(block, layers=[3, 4, 6, 3], plane_num_classes=plane_num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, segment_num_classes=segment_num_classes, output_stride=output_stride,
                 norm_layer=None)

def deeplabv3plus_resnet101(block=AttentionBottleneck,  plane_num_classes=11,
                 segment_num_classes=24, output_stride=8):
    return ResNetDeepLab_plusv2(block, layers=[3, 4, 23, 3], plane_num_classes=plane_num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, segment_num_classes=segment_num_classes, output_stride=output_stride,
                 norm_layer=None)

def deeplabv3plus_resnext50_32x4d(block=AttentionBottleneck,  plane_num_classes=11,
                 segment_num_classes=24, output_stride=8):
    return ResNetDeepLab_plusv2(block, layers=[3, 4, 6, 3], plane_num_classes=plane_num_classes, zero_init_residual=False,
                 groups= 32, width_per_group= 4, segment_num_classes=segment_num_classes, output_stride=output_stride,
                 norm_layer=None)

def deeplabv3plus_resnext50_32x4d_plus(block=AttentionBottleneck,  plane_num_classes=11,
                 segment_num_classes=24, output_stride=8):
    return ResNetDeepLab_plus(block, layers=[3, 4, 6, 3], plane_num_classes=plane_num_classes, zero_init_residual=False,
                 groups= 32, width_per_group= 4, segment_num_classes=segment_num_classes, output_stride=output_stride,
                 norm_layer=None)


def deeplabv3plus_resnext101_32x8d(block=AttentionBottleneck,  plane_num_classes=10,
                 segment_num_classes=24, output_stride=8):
    return ResNetDeepLab(block, layers=[3, 4, 23, 3], plane_num_classes=plane_num_classes, zero_init_residual=False,
                 groups= 32, width_per_group= 8, segment_num_classes=segment_num_classes, output_stride=output_stride,
                 norm_layer=None)


def make_model(args, parent=False):
    #return  deeplabv3plus_resnext50_32x4d(segment_num_classes=args.num_class)   #

    return deeplabv3plus_resnet101(segment_num_classes=args.num_class)

if __name__ == '__main__':
    import torch
    model = deeplabv3plus_resnet101(block=AttentionBottleneck,  plane_num_classes=11,
                 segment_num_classes=24, output_stride=8).cuda()
    model.eval()
    x = torch.randn([1, 1, 224, 224]).cuda()
    print(model)

    print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (
            sum(p.numel() for p in model.parameters()) / 1000000.0))
    out = model(x)

    print(out[0].shape," out[0].shape ",out[3].shape," out[1].shape ")



    '''
    deeplabv3plus_resnet50
    [ =======> Total params: 39.94M <======= ]
    torch.Size([10, 256, 80, 80]) layer1.shape
    torch.Size([10, 512, 40, 40]) layer2.shape
    torch.Size([10, 1024, 40, 40]) layer3.shape
    torch.Size([10, 2048, 40, 40]) layer4.shape
    torch.Size([10, 11]) class_output.shape
    torch.Size([10, 48, 80, 80]) low_level_feature.shape
    torch.Size([10, 256, 40, 40]) output_feature.shape
    torch.Size([10, 256, 80, 80]) output_feature.shape
    torch.Size([10, 24, 320, 320]) segment_output.shape
    torch.Size([10, 11])  out[0].shape  torch.Size([10, 24, 320, 320])  out[1].shape 

    deeplabv3plus_resnext50_32x4d
    [ =======> Total params: 39.88M <======= ]
    torch.Size([10, 256, 80, 80]) layer1.shape
    torch.Size([10, 512, 40, 40]) layer2.shape
    torch.Size([10, 1024, 40, 40]) layer3.shape
    torch.Size([10, 2048, 40, 40]) layer4.shape
    torch.Size([10, 11]) class_output.shape
    torch.Size([10, 48, 80, 80]) low_level_feature.shape
    torch.Size([10, 256, 40, 40]) output_feature.shape
    torch.Size([10, 256, 80, 80]) output_feature.shape
    torch.Size([10, 24, 320, 320]) segment_output.shape
    torch.Size([10, 11])  out[0].shape  torch.Size([10, 24, 320, 320])  out[1].shape 
    
    '''
