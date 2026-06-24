from typing import Any
import torch.nn as nn

import torch

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Merge_Run_dual(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1):
        '''
        in_channels = out_channel  
        恒等卷积  
        '''
        super(Merge_Run_dual, self).__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4),
            nn.ReLU(inplace=True)
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, ksize, stride, 1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.body(x)
        out = self.relu(out + x)
        return out
    
    
class EResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        self.relu  = nn.ReLU()
        
    def forward(self, x):
        out = self.body(x)
        out = self.relu(out + x)
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1 = nn.Sequential(nn.Conv2d(channel , channel // reduction, 1, 1, 0),
                                nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(channel // reduction, channel , 1, 1, 0),
                                nn.Sigmoid())
    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2


class Block(nn.Module):
    def __init__(self,in_channel,out_channel,group=1) -> None:
        super(Block,self).__init__()
        self.r1 = Merge_Run_dual(in_channel,out_channel)
        self.r2 = ResidualBlock(out_channel,out_channel)
        self.r3 = EResidualBlock(out_channel,out_channel)
        self.ca = CALayer(out_channel)
        
    def forward(self,x):
        r1 = self.r1(x)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        out = self.ca(r3)
        return out 
        

class RIDNet(nn.Module):
    def __init__(self) -> None:
        super(RIDNet,self).__init__()
        inc = 1
        n_feats = 64
        ker_size = 3
        reduction = 16
        
        self.head = nn.Sequential(nn.Conv2d(inc,n_feats,ker_size,stride=1,padding=1),
                                  nn.ReLU(inplace=True))
        self.b1 = Block(n_feats, n_feats)
        self.b2 = Block(n_feats, n_feats)
        self.b3 = Block(n_feats, n_feats)
        self.b4 = Block(n_feats, n_feats)
        
        self.tail = nn.Conv2d(n_feats,inc,ker_size,1,1,groups=1)
        
        self.apply(self._init_weights)
        
        
    def forward(self,x):
        h = self.head(x)
        #print(h.shape)
        
        b1 = self.b1(h)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b_out = self.b4(b3)
        #print(b_out.shape)

        res = self.tail(b_out)
        #print(res.shape)
        f_out = res + x
        return f_out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    


def make_model(args):
    return RIDNet()

if __name__ == '__main__':    
    import thop
    from thop import clever_format
    from thop import profile
    import torch
    input = torch.randn([1, 1, 592, 720])
    
    model = make_model(1)
    out = model(input)
    print(out.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops : {}'.format(flops))
    print('params : {}'.format(params))
    print('Total params: %.4fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
     
# flops : 637.219G
# params : 1.497M
# Total params: 1.4970M