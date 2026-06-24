
import torch.nn.functional as F
import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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
    
    
    
    
class NAFBlock2(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                                bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        #self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        #self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        #self.norm2 = LayerNorm2d(c)


        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        #self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
        #self.act = nn.LeakyReLU()

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.sg(x)
        #x = self.act(x)
        
        x = x * self.sca(x)
        x = self.conv3(x)

        y = inp + x * self.beta

        #x = self.conv4(self.norm2(y))
        #x = self.conv4(y)
        #x = self.sg(x)
        #x = self.act(x)
        #x = self.conv5(x)

        #return y + x * self.gamma
        return y
    
    
    
    
    
    


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(channels)
        

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.flatten(2).permute(0,2,1)
        y = self.norm(x).permute(0,1,2).reshape(B,C,H,W).contiguous()
        
        return y

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

    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                                bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class DWConvBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kersize,dilation=3):
        '''
        this block is a single Equal Conv and Relu with no Group and diliation
        :param input_channel:
        :param output_channel:
        :param kersize:
        '''
        super(DWConvBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=kersize,padding=(((dilation-1) * (kersize - 1)+kersize) // 2),dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel,output_channel,1,1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        out = self.block(x)
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
            layers.append(block2(output_channel))
        self.raiselayer =  nn.Sequential(*layers)
        
    def forward(self,x):
        out = self.raiselayer(x)
        return out


class UnetModule4_1(nn.Module):
    def __init__(self, input_channels,output_channels, Layer,block1,block2, num_blocks,kersize, nb_filter):
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
        self.end = nn.Conv2d(nb_filter[0], output_channels,1,1)

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
    def __init__(self, input_channels,output_channels, Layer,block,block2, num_blocks,kersize, nb_filter):
        '''
        num_class : 类别数量
        input_channels ： 输入的通道数量
        block :基本模块，要保证除了通道数量改变，特征图分辨率不改变
        num_blocks ： 模块的数量
        nb_filter ：就是 模块的通道数量
        deep_supervision  ：深度监督控制开关
        '''
        super(UnetModule5, self).__init__()
        self.layer0_0 = Layer(input_channels, nb_filter[0], block,block2,kersize,num_blocks[0])
        self.down1 = nn.AvgPool2d(2)
       
        self.layer1_0 =Layer(nb_filter[0], nb_filter[1], block,block2, kersize,num_blocks[1])
        self.down2 = nn.AvgPool2d(2)
        
        self.layer2_0 = Layer(nb_filter[1], nb_filter[2], block,block2, kersize,num_blocks[2])
        self.down3 = nn.AvgPool2d(2)
        
        self.layer3_0 = Layer(nb_filter[2], nb_filter[3], block,block2, kersize,num_blocks[3])
        self.down4 = nn.AvgPool2d(2)
        
        self.layer4_0 = Layer(nb_filter[3], nb_filter[4], block,block2, kersize,num_blocks[4])
        self.up4 = UpCat(nb_filter[4])
        
        self.layer3_1 = Layer(nb_filter[3], nb_filter[3], block,block2, kersize,num_blocks[3])
        self.up3 = UpCat(nb_filter[3])
        # - - - - - - - - - - - -
        self.layer2_1 = Layer(nb_filter[2], nb_filter[2], block, block2,kersize,num_blocks[2])
        self.up2 = UpCat(nb_filter[2])
        self.layer1_1 = Layer(nb_filter[1], nb_filter[1], block,block2, kersize,num_blocks[1])
        self.up1 = UpCat(nb_filter[1])
        self.layer0_1 = Layer(nb_filter[0], nb_filter[0], block,block2,kersize,num_blocks[0])
        self.end = nn.Conv2d(nb_filter[0], output_channels,1,1)

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





class NAFNet2(nn.Module):
    def __init__(self,inc,ouc,Layer=BaseLayer2,block1=DWConvBlock,block2 = NAFBlock2,kersize=3):
        super().__init__()
        
        
        nb_filter = [8,16,32,64,128]
        num_blocks = [1,2,2,4,8]
        
        self.UNet = UnetModule5(input_channels=inc,output_channels=ouc,Layer=Layer,block=block1,block2= block2,num_blocks=num_blocks,kersize=kersize,nb_filter=nb_filter)
    
    def forward(self,x):
        out = self.UNet(x)
    
        return out
    
    
def make_model(args):
    model = NAFNet2(1,1,Layer=BaseLayer2,block1=DWConvBlock,block2 = NAFBlock2,kersize=3)
    return model

if __name__ == '__main__':
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    import thop
    from thop import clever_format
    from thop import profile
    import time
    model = make_model(1).cuda()
    print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (sum(p.numel() for p in model.parameters())/1000000.0)) 
    print(model)
    input = torch.randn([1, 1, 576, 768]).cuda()
    st = time.time()
    out = model(input)
    end = time.time()
    print("inference time {:.4f}".format(end-st))
    print(out.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops : {}'.format(flops))
    print('params : {}'.format(params))