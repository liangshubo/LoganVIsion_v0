# -*- coding: UTF-8 -*-
"""
Create on 2023-7-18
@Author: LiangShubo
@email: liangshubo@neusoftmedical.com

"""
import torch.nn.functional as F
import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class CA(nn.Module):
    def __init__(self, channel, reduction=1):
        super(CA, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y





## Residual Channel Attention Block (RCAB)
class RCABlock(nn.Module):
    def __init__(
        self,in_channel, out_channel,kernel_size, reduction=4,
        bias=True, res_scale=1):

        super(RCABlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(in_channel, in_channel, kernel_size=5,padding=2, bias=bias))

            if i == 0:
                modules_body.append(nn.InstanceNorm2d(in_channel))
                modules_body.append(nn.LeakyReLU(inplace=True))

        modules_body.append(CA(in_channel, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        #print(x.shape)
        res = self.body(x)

        #print(res.shape)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class FCN(nn.Module):
    def __init__(self,inc):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(inc, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, inc, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.fcn(x)

class UpCat(nn.Module):
    def __init__(self, in_ch):
        super(UpCat, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch,in_ch//2, 2, stride=2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = x2 + x1
        return x
class BaseBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kersize):
        '''
        this block is a single Equal Conv and Relu with no Group and diliation
        :param input_channel:
        :param output_channel:
        :param kersize:
        '''
        super(BaseBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=kersize,padding=(kersize//2)),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        out = self.block(x)
        return out
    
    
class ResIdentityBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ker_size):
        super().__init__()
        self.conv =  nn.Conv2d(in_channels,out_channels,ker_size,stride=1,padding=(ker_size//2))
    def forward(self,x):
        out = self.conv(x)
        out = out + x
        return out

class RFDBBlock(nn.Module):
    '''
    输入输出维度全部一致 
    '''
    def __init__(self, in_channels,out_channels, distillation_rate=0.5):
        super(RFDBBlock, self).__init__()
        self.dc = in_channels // 2
        self.rc = in_channels
        self.c1_d = nn.Conv2d(in_channels,self.dc,kernel_size=1)
        self.c1_r = ResIdentityBlock(in_channels,self.rc,ker_size=3)
        self.c2_d = nn.Conv2d(self.rc, self.dc, kernel_size=1)
        self.c2_r = ResIdentityBlock(self.rc, self.rc, ker_size=3)
        self.c3_d = nn.Conv2d(self.rc, self.dc, kernel_size=1)
        self.c3_r = ResIdentityBlock(self.rc, self.rc, ker_size=3)
        self.c4 = nn.Conv2d(self.rc, self.dc,kernel_size=3,stride=1,padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.5, inplace=True)
        self.c5  = nn.Conv2d(self.dc * 4,in_channels,kernel_size=1)
    def forward(self,input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)
        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        return out


class DWConvBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kersize):
        '''
        this block is a single Equal Conv and Relu with no Group and diliation
        :param input_channel:
        :param output_channel:
        :param kersize:
        '''
        super(DWConvBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=kersize,padding=(kersize//2),groups=input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel,output_channel,1,1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        out = self.block(x)
        return out


class BaseLayer(nn.Module):
    def __init__(self,input_channel,output_channel,block,kersize,num_block):
        super(BaseLayer,self).__init__()
        '''
        This Basylayer is a layer that connect a block one by one  with only one path
        :param input_channel:
        :param output_channel:
        :param block:
        :param num_block:
        '''
        layers = []
        layers.append(block(input_channel, output_channel,kersize))
        for i in range(num_block - 1):
            layers.append(block(output_channel, output_channel,kersize))
        self.raiselayer =  nn.Sequential(*layers)
    def forward(self,x):
        out = self.raiselayer(x)
        return out

class BaseResLayer(nn.Module):
    def __init__(self,input_channel,output_channel,block,kersize,num_block):
        super(BaseResLayer,self).__init__()
        '''
        This Basylayer is a layer that connect a block one by one  with only one path
        :param input_channel:
        :param output_channel:
        :param block:
        :param num_block:
        '''
        layers = []
        layers.append(block(input_channel, output_channel,kersize))
        for i in range(num_block - 1):
            layers.append(block(output_channel, output_channel,kersize))
        self.raiselayer =  nn.Sequential(*layers)
        if input_channel != output_channel:
            self.identity = nn.Conv2d(input_channel,output_channel,kernel_size=1)
        else:
            self.identity =None
    def forward(self,x):
        if self.identity:
            identity = self.identity(x)
        else:
            identity = x
        out = self.raiselayer(x)
        out = out + identity
        return out

class BaseResLayer_2block(nn.Module):
    def __init__(self,input_channel,output_channel,block1,block2,kersize,num_block):
        super(BaseResLayer_2block,self).__init__()
        '''
        This Basylayer is a layer that connect a block one by one  with only one path
        :param input_channel:
        :param output_channel:
        :param block:
        :param num_block:
        '''
        layers = []
        layers.append(block1(input_channel, output_channel,kersize))
        for i in range(num_block - 1):
            layers.append(block2(output_channel, output_channel,kersize))
        self.raiselayer =  nn.Sequential(*layers)
        if input_channel != output_channel:
            self.identity = nn.Conv2d(input_channel,output_channel,kernel_size=1)
        else:
            self.identity =None
    def forward(self,x):
        if self.identity:
            identity = self.identity(x)
        else:
            identity = x
        out = self.raiselayer(x)
        #print(out.shape)
        out = out + identity
        return out




class BaseLayer2(nn.Module):
    def __init__(self,input_channel,output_channel,block1,block2,kersize,num_block):
        super(BaseLayer2,self).__init__()
        '''
        This Basylayer is a layer that connect two block one by one  with only one path
        block1 turn channel
        block2 identity channel 
        :param input_channel:
        :param output_channel:
        :param block:
        :param num_block:
        '''
        layers = []
        layers.append(block1(input_channel, output_channel,kersize))
        for i in range(num_block - 1):
            layers.append(block2(output_channel, output_channel,kersize))
        self.raiselayer =  nn.Sequential(*layers)
        
    def forward(self,x):
        out = self.raiselayer(x)
        return out



class UnetModule3(nn.Module):
    def __init__(self, input_channels,Layer, block, num_blocks, nb_filter):
        '''
        num_class : 类别数量
        input_channels ： 输入的通道数量
        block :基本模块，要保证除了通道数量改变，特征图分辨率不改变
        num_blocks ： 模块的数量
        nb_filter ：就是 模块的通道数量
        deep_supervision  ：深度监督控制开关
        '''
        super(UnetModule3, self).__init__()
        self.layer0_0 = Layer(input_channels*2, nb_filter[0], block, 3,num_blocks[0])
        self.down1 = nn.AvgPool2d(2)
        #self.layer0_0 = layer(input_channels, nb_filter[0], block, 3,num_blocks[0])
        self.layer1_0 =Layer(nb_filter[0], nb_filter[1], block, 3,num_blocks[1])
        self.down2 = nn.AvgPool2d(2)
        self.layer2_0 = Layer(nb_filter[1], nb_filter[2], block, 3,num_blocks[2])
        self.up2 = UpCat(nb_filter[2])
        self.layer1_1 = Layer(nb_filter[1], nb_filter[1], block, 3,num_blocks[1])
        self.up1 = UpCat(nb_filter[1])
        self.layer0_1 = Layer(nb_filter[0], nb_filter[0], block, 3,num_blocks[0])
        self.end = nn.Conv2d(nb_filter[0], input_channels,1,1)

    def forward(self, input):
        ly0_0 = self.layer0_0(input)
        down1 = self.down1(ly0_0)
        ly1_0 = self.layer1_0(down1)
        down2 = self.down2(ly1_0)
        ly2_0 = self.layer2_0(down2)
        up2 = self.up2(ly2_0,ly1_0)
        ly1_1 = self.layer1_1(up2)
        up1 = self.up1(ly1_1,ly0_0)
        ly0_1 = self.layer0_1(up1)
        out = self.end(ly0_1)
        return  out
    

class UnetModule4(nn.Module):
    def __init__(self, input_channels, Layer,block, num_blocks,kersize, nb_filter):
        '''
        num_class : 类别数量
        input_channels ： 输入的通道数量
        block :基本模块，要保证除了通道数量改变，特征图分辨率不改变
        num_blocks ： 模块的数量
        nb_filter ：就是 模块的通道数量
        deep_supervision  ：深度监督控制开关
        '''
        super(UnetModule4, self).__init__()
        self.layer0_0 = Layer(input_channels, nb_filter[0], block,kersize,num_blocks[0])
        self.down1 = nn.AvgPool2d(2)
        #self.layer0_0 = layer(input_channels, nb_filter[0], block, 3,num_blocks[0])
        self.layer1_0 =Layer(nb_filter[0], nb_filter[1], block, kersize,num_blocks[1])
        self.down2 = nn.AvgPool2d(2)
        self.layer2_0 = Layer(nb_filter[1], nb_filter[2], block, kersize,num_blocks[2])
        # - - - - - - - - - - - -
        self.down3 = nn.AvgPool2d(2)
        self.layer3_0 = Layer(nb_filter[2], nb_filter[3], block, kersize,num_blocks[3])
        self.up3 = UpCat(nb_filter[3])
        # - - - - - - - - - - - -
        self.layer2_1 = Layer(nb_filter[2], nb_filter[2], block, kersize,num_blocks[2])
        self.up2 = UpCat(nb_filter[2])
        self.layer1_1 = Layer(nb_filter[1], nb_filter[1], block, kersize,num_blocks[1])
        self.up1 = UpCat(nb_filter[1])
        self.layer0_1 = Layer(nb_filter[0], nb_filter[0], block,kersize,num_blocks[0])
        self.end = nn.Conv2d(nb_filter[0], input_channels,1,1)

    def forward(self, input):
        ly0_0 = self.layer0_0(input)
        down1 = self.down1(ly0_0)
        ly1_0 = self.layer1_0(down1)
        down2 = self.down2(ly1_0)
        ly2_0 = self.layer2_0(down2)
        # - - - - - - 
        down3 = self.down3(ly2_0)
        ly3_0 = self.layer3_0(down3)
        up3 = self.up3(ly3_0,ly2_0)
        # - - - - - - 
        ly2_1 = self.layer2_1(up3)
        up2 = self.up2(ly2_1,ly1_0)
        ly1_1 = self.layer1_1(up2)
        up1 = self.up1(ly1_1,ly0_0)
        ly0_1 = self.layer0_1(up1)
        out = self.end(ly0_1)
        return  out



class UnetModule5(nn.Module):
    def __init__(self, input_channels, Layer,block, num_blocks,kersize, nb_filter):
        '''
        num_class : 类别数量
        input_channels ： 输入的通道数量
        block :基本模块，要保证除了通道数量改变，特征图分辨率不改变
        num_blocks ： 模块的数量
        nb_filter ：就是 模块的通道数量
        deep_supervision  ：深度监督控制开关
        '''
        super(UnetModule5, self).__init__()
        self.layer0_0 = Layer(input_channels, nb_filter[0], block,kersize,num_blocks[0])
        self.down1 = nn.AvgPool2d(2)
       
        self.layer1_0 =Layer(nb_filter[0], nb_filter[1], block, kersize,num_blocks[1])
        self.down2 = nn.AvgPool2d(2)
        
        self.layer2_0 = Layer(nb_filter[1], nb_filter[2], block, kersize,num_blocks[2])
        self.down3 = nn.AvgPool2d(2)
        
        self.layer3_0 = Layer(nb_filter[2], nb_filter[3], block, kersize,num_blocks[3])
        self.down4 = nn.AvgPool2d(2)
        
        self.layer4_0 = Layer(nb_filter[3], nb_filter[4], block, kersize,num_blocks[4])
        self.up4 = UpCat(nb_filter[4])
        
        self.layer3_1 = Layer(nb_filter[3], nb_filter[3], block, kersize,num_blocks[3])
        self.up3 = UpCat(nb_filter[3])
        # - - - - - - - - - - - -
        self.layer2_1 = Layer(nb_filter[2], nb_filter[2], block, kersize,num_blocks[2])
        self.up2 = UpCat(nb_filter[2])
        self.layer1_1 = Layer(nb_filter[1], nb_filter[1], block, kersize,num_blocks[1])
        self.up1 = UpCat(nb_filter[1])
        self.layer0_1 = Layer(nb_filter[0], nb_filter[0], block,kersize,num_blocks[0])
        self.end = nn.Conv2d(nb_filter[0], input_channels,1,1)

    def forward(self, input):
        ly0_0 = self.layer0_0(input)
        down1 = self.down1(ly0_0)
        ly1_0 = self.layer1_0(down1)
        down2 = self.down2(ly1_0)
        ly2_0 = self.layer2_0(down2)
        # - - - - - - 
        down3 = self.down3(ly2_0)
        ly3_0 = self.layer3_0(down3)
        down4 = self.down4(ly3_0)
        ly4_0 = self.layer4_0(down4)
        
        up4 = self.up4(ly4_0,ly3_0)
        ly3_1 = self.layer3_1(up4)
        
        up3 = self.up3(ly3_1,ly2_0)
        # - - - - - - 
        ly2_1 = self.layer2_1(up3)
        up2 = self.up2(ly2_1,ly1_0)
        ly1_1 = self.layer1_1(up2)
        up1 = self.up1(ly1_1,ly0_0)
        ly0_1 = self.layer0_1(up1)
        out = self.end(ly0_1)
        return  out

class UnetModule5_1(nn.Module):
    def __init__(self, input_channels, Layer,block1,block2, num_blocks,kersize, nb_filter):
        '''
        用于主要模块是恒等的块，在层内只用第一个块进行通道维度括充 ， 其他的块进行恒等处理 ，注意这里的Layer 要用双模块层 
        num_class : 类别数量
        input_channels ： 输入的通道数量
        block :基本模块，要保证除了通道数量改变，特征图分辨率不改变
        num_blocks ： 模块的数量
        nb_filter ：就是 模块的通道数量
        deep_supervision  ：深度监督控制开关
        '''
        super(UnetModule5_1, self).__init__()
        self.layer0_0 = Layer(input_channels, nb_filter[0], block1,block2,kersize,num_blocks[0])
        self.down1 = nn.AvgPool2d(2)
       
        self.layer1_0 =Layer(nb_filter[0], nb_filter[1],block1,block2, kersize,num_blocks[1])
        self.down2 = nn.AvgPool2d(2)
        
        self.layer2_0 = Layer(nb_filter[1], nb_filter[2],block1,block2, kersize,num_blocks[2])
        self.down3 = nn.AvgPool2d(2)
        
        self.layer3_0 = Layer(nb_filter[2], nb_filter[3], block1,block2, kersize,num_blocks[3])
        self.down4 = nn.AvgPool2d(2)
        
        self.layer4_0 = Layer(nb_filter[3], nb_filter[4], block1,block2, kersize,num_blocks[4])
        self.up4 = UpCat(nb_filter[4])
        
        self.layer3_1 = Layer(nb_filter[3], nb_filter[3], block1,block2, kersize,num_blocks[3])
        self.up3 = UpCat(nb_filter[3])
        # - - - - - - - - - - - -
        self.layer2_1 = Layer(nb_filter[2], nb_filter[2], block1,block2, kersize,num_blocks[2])
        self.up2 = UpCat(nb_filter[2])
        self.layer1_1 = Layer(nb_filter[1], nb_filter[1], block1,block2, kersize,num_blocks[1])
        self.up1 = UpCat(nb_filter[1])
        self.layer0_1 = Layer(nb_filter[0], nb_filter[0], block1,block2,kersize,num_blocks[0])
        self.end = nn.Conv2d(nb_filter[0], input_channels,1,1)

    def forward(self, input):

        ly0_0 = self.layer0_0(input)
        down1 = self.down1(ly0_0)
        ly1_0 = self.layer1_0(down1)
        down2 = self.down2(ly1_0)
        ly2_0 = self.layer2_0(down2)
        # - - - - - - 
        down3 = self.down3(ly2_0)
        ly3_0 = self.layer3_0(down3)
        down4 = self.down4(ly3_0)
        ly4_0 = self.layer4_0(down4)
        
        up4 = self.up4(ly4_0,ly3_0)
        ly3_1 = self.layer3_1(up4)
        
        up3 = self.up3(ly3_1,ly2_0)
        # - - - - - - 
        ly2_1 = self.layer2_1(up3)
        up2 = self.up2(ly2_1,ly1_0)
        ly1_1 = self.layer1_1(up2)
        up1 = self.up1(ly1_1,ly0_0)
        ly0_1 = self.layer0_1(up1)
        out = self.end(ly0_1)
        return  out



class UnetModule4_1(nn.Module):
    def __init__(self, input_channels, Layer,block1,block2, num_blocks,kersize, nb_filter):
        '''
        num_class : 类别数量
        input_channels ： 输入的通道数量
        block :基本模块，要保证除了通道数量改变，特征图分辨率不改变
        num_blocks ： 模块的数量
        nb_filter ：就是 模块的通道数量
        deep_supervision  ：深度监督控制开关
        '''
        super(UnetModule4_1, self).__init__()
        self.layer0_0 = Layer(input_channels, nb_filter[0], block1,block2,kersize,num_blocks[0])
        self.down1 = nn.AvgPool2d(2)
        #self.layer0_0 = layer(input_channels, nb_filter[0], block, 3,num_blocks[0])
        self.layer1_0 =Layer(nb_filter[0], nb_filter[1], block1,block2, kersize,num_blocks[1])
        self.down2 = nn.AvgPool2d(2)
        self.layer2_0 = Layer(nb_filter[1], nb_filter[2], block1,block2, kersize,num_blocks[2])
        # - - - - - - - - - - - -
        self.down3 = nn.AvgPool2d(2)
        self.layer3_0 = Layer(nb_filter[2], nb_filter[3], block1,block2, kersize,num_blocks[3])
        self.up3 = UpCat(nb_filter[3])
        # - - - - - - - - - - - -
        self.layer2_1 = Layer(nb_filter[2], nb_filter[2], block1,block2, kersize,num_blocks[2])
        self.up2 = UpCat(nb_filter[2])
        self.layer1_1 = Layer(nb_filter[1], nb_filter[1], block1,block2, kersize,num_blocks[1])
        self.up1 = UpCat(nb_filter[1])
        self.layer0_1 = Layer(nb_filter[0], nb_filter[0], block1,block2,kersize,num_blocks[0])
        self.end = nn.Conv2d(nb_filter[0], input_channels,1,1)

    def forward(self, input):
        ly0_0 = self.layer0_0(input)
        down1 = self.down1(ly0_0)
        ly1_0 = self.layer1_0(down1)
        down2 = self.down2(ly1_0)
        ly2_0 = self.layer2_0(down2)
        # - - - - - - 
        down3 = self.down3(ly2_0)
        ly3_0 = self.layer3_0(down3)
        up3 = self.up3(ly3_0,ly2_0)
        # - - - - - - 
        ly2_1 = self.layer2_1(up3)
        up2 = self.up2(ly2_1,ly1_0)
        ly1_1 = self.layer1_1(up2)
        up1 = self.up1(ly1_1,ly0_0)
        ly0_1 = self.layer0_1(up1)
        out = self.end(ly0_1)
        return  out



class baselinecbdnet(nn.Module):
    def __init__(self,inc) -> None:
        super().__init__()
        nb_filter = [64, 128, 256]
        num_blocks = [2, 3, 6]
        self.fcn = FCN(inc=inc)
        self.Unet = UnetModule3(input_channels=inc,Layer=BaseLayer,block=BaseBlock,num_blocks=num_blocks,nb_filter=nb_filter)
    def forward(self,x):
        noise_es = self.fcn(x)
        concat = torch.cat([x,noise_es],dim=1)
        clean = self.Unet(concat)+x
        return clean
class baselineUnet(nn.Module):
    def __init__(self,inc,Layer=BaseLayer,block=DWConvBlock,kersize=3) -> None:
        super().__init__()
        nb_filter = [32,64,128,256]
        num_blocks = [1,2,3,6]
        self.UNet = UnetModule4(input_channels=inc,Layer=Layer,block=block,num_blocks=num_blocks,kersize=kersize,nb_filter=nb_filter)
    def forward(self,x):
        out = self.UNet(x)
        out = x + out
        return out
    
class baselineUnet2(nn.Module):
    #减少低级信息处理流程  
    def __init__(self,inc,Layer=BaseLayer,block=DWConvBlock,kersize=3) -> None:
        super().__init__()
        nb_filter = [32,64,128,256]
        num_blocks = [1,1,3,7]
        self.UNet = UnetModule4(input_channels=inc,Layer=Layer,block=block,num_blocks=num_blocks,kersize=kersize,nb_filter=nb_filter)
    def forward(self,x):
        out = self.UNet(x)
        out = x + out
        return out
    

class baselineUnetRFDN(nn.Module):
    def __init__(self,inc,Layer=BaseLayer2,block1=BaseBlock,block2 = RFDBBlock,kersize = 3) -> None:
        super().__init__()
        nb_filter = [32,64,128,256]
        num_blocks = [1,2,3,6]
        self.UNet = UnetModule4_1(input_channels=inc,Layer=Layer,block1=block1,block2= block2,num_blocks=num_blocks,kersize=kersize,nb_filter=nb_filter)
    def forward(self,x):
        out = self.UNet(x)
        out = x + out
        return out
    
class baselineUnet_light(nn.Module):
    def __init__(self,inc,Layer=BaseLayer,block=DWConvBlock,kersize=3) -> None:
        super().__init__()
        nb_filter = [8,16,32,64,128]
        num_blocks = [1,1,2,2,4]
        # avg 1.52 ms
        # light 1.44 ms
        self.UNet = UnetModule5(input_channels=inc,Layer=Layer,block=block,num_blocks=num_blocks,kersize=kersize,nb_filter=nb_filter)
    def forward(self,x):
        out = self.UNet(x)
        out = x + out
        return out
    


    
class baselineUnet_mid_diff(nn.Module):
    def __init__(self,inc,Layer=BaseResLayer_2block,block1=BaseBlock,block2 = RCABlock,kersize=3) -> None:
        super().__init__()
        nb_filter = [8,16,32,64,128]
        num_blocks = [2,2,2,2,4]
        # avg 1.52 ms
        # light 1.44 ms
        self.UNet = UnetModule5_1(input_channels=inc,Layer=Layer,block1=block1,block2=block2,num_blocks=num_blocks,kersize=kersize,nb_filter=nb_filter)
    def forward(self,x,t):
        out = self.UNet(x)
        out = x + out
        return out
    
    
def make_model(args):
    #model =baselineUnet(1,Layer=BaseLayer,block=BaseBlock,kersize=3) # EX3   kersize =3 .EX4 kersize = 7 ;; 8EX2
    # model = baselineUnetRFDN(1,Layer=BaseLayer2,block1=BaseBlock,block2 = RFDBBlock,kersize=3) 
    
    
    
    model =baselineUnet_mid(1) # EX1，EX2,   8EX1,8EX2   # Light_Unet  1,Layer=BaseLayer,block=BaseBlock,kersize=3
    
    #model = baselineUnet(1,Layer=BaseResLayer,block=BaseBlock,kersize=3)   #8ex3
    return model

if __name__ == '__main__':
    import thop
    import time
    from thop import clever_format
    from thop import profile
    model = make_model(1).cuda()
    print(model)
    input = torch.randn([1, 1, 592, 720]).cuda()

    start = time.time()
    out = model(input)
    end = time.time()
    print(f"time cost {end - start}")
    print(out.shape)
    print(out.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops : {}'.format(flops))
    print('params : {}'.format(params))

    tt=0
    for i in range(100):
        start = time.time()
        out = model(input)
        end = time.time()
       # print(f"time cost {(end - start)*1000:.2f}ms num: {i}")
        tt += end - start
    print(f"avg {tt*10:.2f} ms")
'''
flops : 77.008G
params : 4.374M
'''

'''
model =baselineUnet(1,Layer=BaseResLayer,block=BaseBlock,kersize=3) 
flops : 72.309G
params : 4.412M
'''

'''
model =baselineUnet(1,Layer=BaseResLayer,block=DWConvBlock,kersize=3).
flops : 20.007G
params : 754.625K
'''