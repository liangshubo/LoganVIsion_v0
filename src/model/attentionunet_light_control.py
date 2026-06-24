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
from torch.nn.parameter import Parameter


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



class Attention(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel,outchannel, groups=8):
        super(Attention, self).__init__()
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
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w) # 按照通道维度进行分组 ，所以要保证通道数量被组数量整除  
        x_0, x_1 = x.chunk(2, dim=1)
        # channel attention
        xn = self.avg_pool(x_0)
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


class AttentionBlock(nn.Module):
    def __init__(self,inchannel,outchannel, kersize, groups=8) -> None:
        super(AttentionBlock,self).__init__()
        
        self.Conv1 = nn.Conv2d(inchannel,outchannel,kernel_size=kersize,stride=1,padding=int(kersize//2))
        self.relu = nn.ReLU()
        self.sa = Attention(inchannel,outchannel,groups=groups)    
        self.Conv2 = nn.Conv2d(inchannel,outchannel,kernel_size=1)
    def forward(self,x):
        res = x
        out = self.relu(self.Conv1(x))
        #print(out.shape)
        out = self.sa(out)
        out = self.Conv2(out)
        out = res + x  
        return out  



class AttentionControlBlock(nn.Module):
    def __init__(self,inchannel,outchannel, kersize, groups=8) -> None:
        super(AttentionControlBlock,self).__init__()
        
        self.Conv1 = nn.Conv2d(inchannel,outchannel,kernel_size=kersize,stride=1,padding=int(kersize//2))
        self.relu = nn.ReLU()
        self.Conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=1)
        
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spacil = nn.Conv2d(outchannel,1,kernel_size=1)
        #self.cweight = Parameter(torch.zeros(1, outchannel, 1, 1))
        #self.cbias = Parameter(torch.ones(1, outchannel , 1, 1))
        #self.sweight = Parameter(torch.zeros(1, outchannel , 1, 1))
        #self.sbias = Parameter(torch.ones(1, outchannel , 1, 1))
        self.sigmoid = nn.Sigmoid()
        # self.Conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=1)
        
    def forward(self,x):
        out = self.Conv2(self.relu(self.Conv1(x)))
        #print(out.shape)
        
        xn = self.avg_pool(out)
        # xn = self.cweight * xn + self.cbias
        xn = self.sigmoid(xn)
        
        xs = self.spacil(out)
        xs = self.sigmoid(xs)
        
        return out  ,xn,xs




class ResIdentityBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ker_size):
        super().__init__()
        self.conv =  nn.Conv2d(in_channels,out_channels,ker_size,stride=1,padding=(ker_size//2))
    def forward(self,x):
        out = self.conv(x)
        out = out + x
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
        
        #out = out*xn + xs
        return out


class BaseLayer2(nn.Module):
    def __init__(self,input_channel,output_channel,block1,kersize,num_block):
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
            layers.append(block1(output_channel))
        self.raiselayer =  nn.Sequential(*layers)
        
    def forward(self,x):
        out = self.raiselayer(x)
        return out


class TwoBlockResLayer(nn.Module):
    def __init__(self,input_channel,output_channel,block,block2,kersize,num_block):
        super(TwoBlockResLayer,self).__init__()
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
            layers.append(block2(output_channel, output_channel,kersize))
            
            
        self.raiselayer =  nn.Sequential(*layers)
        if input_channel != output_channel:
            self.identity = nn.Conv2d(input_channel,output_channel,kernel_size=1)
        else:
            self.identity =None
    def forward(self,x,xn,xs):
        if self.identity:
            identity = self.identity(x)
        else:
            identity = x
        out = self.raiselayer(x)
        out = out + identity
        
        out = out*xn+xs
        return out






class UnetModule5_control(nn.Module):
    def __init__(self, input_channels,output_channels, Layer,Layer2,block,block2,block3, num_blocks,kersize, nb_filter):
        '''
        num_class : 类别数量
        input_channels ： 输入的通道数量
        block :基本模块，要保证除了通道数量改变，特征图分辨率不改变
        num_blocks ： 模块的数量
        nb_filter ：就是 模块的通道数量
        deep_supervision  ：深度监督控制开关
        '''
        super(UnetModule5_control, self).__init__()
        self.layer0_0 = Layer(input_channels, nb_filter[0], block,block,kersize,num_blocks[0])
        self.fc0_0 = Layer2(input_channels, nb_filter[0], block3,kersize,num_blocks[0])    
        self.down1 = nn.AvgPool2d(2)
       
        self.layer1_0 =Layer(nb_filter[0], nb_filter[1], block,block, kersize,num_blocks[1])
        self.fc1_0 = Layer2(nb_filter[0], nb_filter[1], block3, kersize,num_blocks[0])   
        self.down2 = nn.AvgPool2d(2)
        
        self.layer2_0 = Layer(nb_filter[1], nb_filter[2], block,block, kersize,num_blocks[2])
        self.fc2_0 = Layer2(nb_filter[1], nb_filter[2], block3, kersize,num_blocks[0])
        self.down3 = nn.AvgPool2d(2)
        
        
        self.layer3_0 = Layer(nb_filter[2], nb_filter[3], block,block2, kersize,num_blocks[3])
        self.fc3_0  = Layer2(nb_filter[2], nb_filter[3], block3, kersize,num_blocks[0])
        self.down4 = nn.AvgPool2d(2)
        
        self.layer4_0 = Layer(nb_filter[3], nb_filter[4], block,block2, kersize,num_blocks[4])
        self.fc4_0 =  Layer2(nb_filter[3], nb_filter[4], block3, kersize,num_blocks[0])
        self.up4 = UpCat(nb_filter[4])
        
        self.layer3_1 = Layer(nb_filter[3], nb_filter[3], block,block2, kersize,num_blocks[3])
        self.fc3_1 =  Layer2(nb_filter[3], nb_filter[3], block3, kersize,num_blocks[0])
        self.up3 = UpCat(nb_filter[3])
        # - - - - - - - - - - - -
        self.layer2_1 = Layer(nb_filter[2], nb_filter[2], block, block,kersize,num_blocks[2])
        self.fc2_1 =  Layer2(nb_filter[2], nb_filter[2], block3,kersize,num_blocks[0])
        self.up2 = UpCat(nb_filter[2])
        
        self.layer1_1 = Layer(nb_filter[1], nb_filter[1], block,block, kersize,num_blocks[1])
        self.fc1_1 = Layer2(nb_filter[1], nb_filter[1], block3, kersize,num_blocks[0])
        self.up1 = UpCat(nb_filter[1])
        
        
        self.layer0_1 = Layer(nb_filter[0], nb_filter[0], block,block,kersize,num_blocks[0])
        self.fc0_1 = Layer2(nb_filter[0], nb_filter[0], block3,kersize,num_blocks[0])
        
        # -------- 
        self.end = nn.Conv2d(nb_filter[0], output_channels,1,1)
        
        

    def forward(self, input,idx):
        idx_input = torch.ones_like(input)*idx
        
        fc0_0,fc0_0xn,fc0_0xs = self.fc0_0(idx_input)
        ly0_0 = self.layer0_0(input,fc0_0xn,fc0_0xs)
        
        
        down1 = self.down1(ly0_0)
        fdown1 = self.down1(fc0_0)
        
        
        fc1_0,fc1_0xn,fc1_0xs = self.fc1_0(fdown1)
        ly1_0 = self.layer1_0(down1,fc1_0xn,fc1_0xs)
        
        
        down2 = self.down2(ly1_0)
        fdown2 = self.down2(fc1_0)
        
        
        fc2_0,fc2_0xn,fc2_0xs = self.fc2_0(fdown2)
        ly2_0 = self.layer2_0(down2,fc2_0xn,fc2_0xs)
        
        
        # - - - - - - 
        down3 = self.down3(ly2_0)
        fdown3 = self.down3(fc2_0)
        
        fc3_0,fc3_0xn,fc3_0xs = self.fc3_0(fdown3)
        ly3_0 = self.layer3_0(down3,fc3_0xn,fc3_0xs)
        
        down4 = self.down4(ly3_0)
        fdown4 = self.down4(fc3_0)
        
        fc4_0,fc4_0xn,fc4_0xs = self.fc4_0(fdown4)
        ly4_0 = self.layer4_0(down4,fc4_0xn,fc4_0xs)
        
        
        up4 = self.up4(ly4_0,ly3_0)
        fup4 = self.up4(fc4_0,fc3_0)
        
        fc3_1,fc3_1xn,fc3_1xs = self.fc3_1(fup4)
        ly3_1 = self.layer3_1(up4,fc3_1xn,fc3_1xs)
        
        up3 = self.up3(ly3_1,ly2_0)
        fup3 = self.up3(fc3_1,fc2_0)
        # - - - - - - 
        fc2_1,fc2_1xn,fc2_1xs = self.fc2_1(fup3)
        ly2_1 = self.layer2_1(up3,fc2_1xn,fc2_1xs)
        
        up2 = self.up2(ly2_1,ly1_0)
        fup2 = self.up2(fc2_1,fc1_0)
        
        fc1_1,fc1_1xn,fc1_1xs = self.fc1_1(fup2)
        ly1_1 = self.layer1_1(up2,fc1_1xn,fc1_1xs)
        
        
        up1 = self.up1(ly1_1,ly0_0)
        fup1 = self.up1(fc1_1,fc0_0)
        
        fc0_1,fc0_1xn,fc0_1xs = self.fc0_1(fup1)
        
        
        ly0_1 = self.layer0_1(up1,fc0_1xn,fc0_1xs)
        out = self.end(ly0_1)
        return  out


class UnetModule4_2Block(nn.Module):
    def __init__(self, input_channels, Layer,block, num_blocks,kersize, nb_filter, block2=None):
        '''
        num_class : 类别数量
        input_channels ： 输入的通道数量
        block :基本模块，要保证除了通道数量改变，特征图分辨率不改变
        num_blocks ： 模块的数量
        nb_filter ：就是 模块的通道数量
        deep_supervision  ：深度监督控制开关
        '''
        super(UnetModule4_2Block, self).__init__()
        self.layer0_0 = Layer(input_channels, nb_filter[0], block,block2,kersize,num_blocks[0])
        self.down1 = nn.AvgPool2d(2)
        #self.layer0_0 = layer(input_channels, nb_filter[0], block, 3,num_blocks[0])
        self.layer1_0 =Layer(nb_filter[0], nb_filter[1], block,block2, kersize,num_blocks[1])
        self.down2 = nn.AvgPool2d(2)
        self.layer2_0 = Layer(nb_filter[1], nb_filter[2], block, block2,kersize,num_blocks[2])
        # - - - - - - - - - - - -
        self.down3 = nn.AvgPool2d(2)
        self.layer3_0 = Layer(nb_filter[2], nb_filter[3], block, block2,kersize,num_blocks[3])
        self.up3 = UpCat(nb_filter[3])
        # - - - - - - - - - - - -
        self.layer2_1 = Layer(nb_filter[2], nb_filter[2], block,block2, kersize,num_blocks[2])
        self.up2 = UpCat(nb_filter[2])
        self.layer1_1 = Layer(nb_filter[1], nb_filter[1], block,block2, kersize,num_blocks[1])
        self.up1 = UpCat(nb_filter[1])
        self.layer0_1 = Layer(nb_filter[0], nb_filter[0], block,block2,kersize,num_blocks[0])
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




class AttentionUnet_Light_control(nn.Module):
    def __init__(self,inc,ouc,Layer=TwoBlockResLayer,Layer2=None,block1=BaseBlock, num_blocks =None,kersize=3,nb_filter=None,block2=None,block3=None) -> None:
        super().__init__()
        #nb_filter = [8,16,32,64,128]
        #num_blocks = [1,2,2,4,8]
        
        self.UNet = UnetModule5_control(input_channels=inc,output_channels=ouc,Layer=Layer,Layer2=Layer2,block=block1,block2= block2,block3=block3,num_blocks=num_blocks,kersize=kersize,nb_filter=nb_filter)

   
   
    def forward(self,x,idx):
        out = self.UNet(x,idx)
        out = x + out
        return out
    


    
def make_model(args):
    nb_filter = [8,16,32,64,128]
    num_blocks = [1,1,2,2,4]
    model = AttentionUnet_Light_control(1,1,Layer=TwoBlockResLayer,Layer2 = BaseLayer2,block1=BaseBlock,kersize=3,num_blocks=num_blocks,nb_filter=nb_filter,block2=AttentionBlock,block3=AttentionControlBlock)   #8ex3
    return model

if __name__ == '__main__':
    import thop
    from thop import clever_format
    from thop import profile
    model = make_model(1)
    print(model)
    input = torch.randn([1, 1, 454, 672])

    out = model(input,3)
    print(out.shape)
    flops, params = profile(model, inputs=(input,1,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops : {}'.format(flops))
    print('params : {}'.format(params))

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



# ----------------------------------------
nb_filter = [32,64,128,256]
num_blocks = [1,2,3,6]
model = UnetModule4(1,Layer=BaseResLayer,block=BaseBlock,kersize=3,num_blocks=num_blocks,nb_filter=nb_filter)
flops : 72.309G
params : 4.412M

#----------------------------------------
nb_filter = [32,64,128,256]
num_blocks = [1,2,3,6]

'''