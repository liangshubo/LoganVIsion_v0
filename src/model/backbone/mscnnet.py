# %%
import yaml


import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
#from bricks import DownSample, LayerScale, StochasticDepth, DWConv3x3, NormLayer

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
#from sync_bn.nn.modules import SynchronizedBatchNorm2d
from functools import partial

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class myLayerNorm(nn.Module):
    def __init__(self, inChannels):
        super().__init__()
        self.norm == nn.LayerNorm(inChannels, eps=1e-5)

    def forward(self, x):
        # reshaping only to apply Layer Normalization layer
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B*C*H*W -> B*C*HW -> B*HW*C
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B*HW*C -> B*H*W*C -> B*C*H*W

        return x


class NormLayer(nn.Module):
    def __init__(self, inChannels, norm_type='batch_norm'):
        super().__init__()
        self.inChannels = inChannels
        self.norm_type = norm_type
        if norm_type == 'batch_norm':
            # print('Adding Batch Norm layer') # for testing
            self.norm = nn.BatchNorm2d(inChannels, eps=1e-5)

        elif norm_type == 'layer_norm':
            # print('Adding Layer Norm layer') # for testing
            self.norm == nn.myLayerNorm(inChannels)
        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.norm(x)

        return x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inChannels}, norm_type={self.norm_type})'


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class LayerScale(nn.Module):
    '''
    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    '''

    def __init__(self, inChannels, init_value=1e-2):
        super().__init__()
        self.inChannels = inChannels
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones((inChannels)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1)  # C, -> C,1,1
            return scale * x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inChannels}, init_value={self.init_value})'


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def stochastic_depth(input: torch.Tensor, p: float,
                     mode: str, training: bool = True):
    if not training or p == 0.0:
        # print(f'not adding stochastic depth of: {p}')
        return input

    survival_rate = 1.0 - p
    if mode == 'row':
        shape = [input.shape[0]] + [1] * (input.ndim - 1)  # just converts BXCXHXW -> [B,1,1,1] list
    elif mode == 'batch':
        shape = [1] * input.ndim

    noise = torch.empty(shape, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    # print(f'added sDepth of: {p}')
    return input * noise


class StochasticDepth(nn.Module):
    '''
    Stochastic Depth module.
    It performs ROW-wise dropping rather than sample-wise.
    mode (str): ``"batch"`` or ``"row"``.
                ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                randomly selected rows from the batch.
    References:
      - https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
    '''

    def __init__(self, p=0.5, mode='row'):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p})"
        return s


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None,
           warning=True):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class DownSample(nn.Module):
    def __init__(self, kernelSize=3, stride=2, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(kernelSize, kernelSize),
                              stride=stride, padding=(kernelSize // 2, kernelSize // 2))
        # stride 4 => 4x down sample
        # stride 2 => 2x down sample

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1,2)
        return x, H, W


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DWConv3x3(nn.Module):
    '''Depth wise conv'''

    def __init__(self, dim=768):
        super(DWConv3x3, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class ConvBNRelu(nn.Module):

    @classmethod
    def _same_paddings(cls, kernel):
        if kernel == 1:
            return 0
        elif kernel == 3:
            return 1

    def __init__(self, inChannels, outChannels, kernel=3, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()

        if padding == 'same':
            padding = self._same_paddings(kernel)

        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel,
                              padding=padding, stride=stride, dilation=dilation,
                              groups=groups, bias=False)
        self.norm = NormLayer(outChannels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class SeprableConv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernal_size=3, bias=False):
        self.dwconv = nn.Conv2d(inChannels, inChannels, kernal_size=kernal_size,
                                groups=inChannels, bias=bias)
        self.pwconv = nn.Conv2d(inChannels, inChannels, kernal_size=1, bias=bias)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)

        return x


class ConvRelu(nn.Module):
    def __init__(self, inChannels, outChannels, kernel=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


class StemConv(nn.Module):
    '''following ConvNext paper'''

    def __init__(self, in_channels, out_channels, bn_momentum=0.99):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            NormLayer(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            NormLayer(out_channels)
        )

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        return x, H, W


class FFN(nn.Module):
    '''following ConvNext paper'''

    def __init__(self, in_channels, out_channels, hid_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hid_channels, 1)
        self.dwconv = DWConv3x3(hid_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class BlockFFN(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, ls_init_val=1e-2, drop_path=0.):
        super().__init__()
        self.norm = NormLayer(in_channels)
        self.ffn = FFN(in_channels, out_channels, hid_channels)
        self.layer_scale = LayerScale(in_channels, init_value=ls_init_val)
        self.drop_path = StochasticDepth(p=drop_path)

    def forward(self, x):
        skip = x.clone()

        x = self.norm(x)
        x = self.ffn(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        op = skip + x
        return op


class MSCA(nn.Module):

    def __init__(self, dim):
        super(MSCA, self).__init__()
        # input
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # split into multipats of multiscale attention
        self.conv17_0 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv17_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv111_0 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv111_1 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv211_0 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv211_1 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv11 = nn.Conv2d(dim, dim, 1)  # channel mixer

    def forward(self, x):
        skip = x.clone()

        c55 = self.conv55(x)
        c17 = self.conv17_0(x)
        c17 = self.conv17_1(c17)
        c111 = self.conv111_0(x)
        c111 = self.conv111_1(c111)
        c211 = self.conv211_0(x)
        c211 = self.conv211_1(c211)

        add = c55 + c17 + c111 + c211

        mixer = self.conv11(add)

        op = mixer * skip

        return op


class BlockMSCA(nn.Module):
    def __init__(self, dim, ls_init_val=1e-2, drop_path=0.0):
        super().__init__()
        self.norm = NormLayer(dim)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.msca = MSCA(dim)
        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.layer_scale = LayerScale(dim, init_value=ls_init_val)
        self.drop_path = StochasticDepth(p=drop_path)
        # print(f'BlockMSCA {drop_path}')

    def forward(self, x):
        skip = x.clone()

        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.msca(x)
        x = self.proj2(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        out = x + skip

        return out


class StageMSCA(nn.Module):
    def __init__(self, dim, ffn_ratio=4., ls_init_val=1e-2, drop_path=0.0):
        super().__init__()
        # print(f'StageMSCA {drop_path}')
        self.msca_block = BlockMSCA(dim, ls_init_val, drop_path)

        ffn_hid_dim = int(dim * ffn_ratio)
        self.ffn_block = BlockFFN(in_channels=dim, out_channels=dim,
                                  hid_channels=ffn_hid_dim, ls_init_val=ls_init_val,
                                  drop_path=drop_path)

    def forward(self, x):  # input coming form Stem
        # B, N, C = x.shape
        # x = x.permute()
        x = self.msca_block(x)
        x = self.ffn_block(x)

        return x


class MSCANet(nn.Module):
    def __init__(self, in_channnels=3, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 ls_init_val=1e-2, drop_path=0.0):
        super(MSCANet, self).__init__()
        # print(f'MSCANet {drop_path}')
        self.depths = depths
        self.num_stages = num_stages
        # stochastic depth decay rule (similar to linear decay) / just like matplot linspace
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                input_embed = StemConv(in_channnels, embed_dims[0])
            else:
                input_embed = DownSample(in_channels=embed_dims[i - 1], embed_dim=embed_dims[i])

            stage = nn.ModuleList([StageMSCA(dim=embed_dims[i], ffn_ratio=ffn_ratios[i],
                                             ls_init_val=ls_init_val, drop_path=dpr[cur + j])
                                   for j in range(depths[i])])

            norm_layer = NormLayer(embed_dims[i])
            cur += depths[i]

            setattr(self, f'input_embed{i + 1}', input_embed)
            setattr(self, f'stage{i + 1}', stage)
            setattr(self, f'norm_layer{i + 1}', norm_layer)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            input_embed = getattr(self, f'input_embed{i + 1}')
            stage = getattr(self, f'stage{i + 1}')
            norm_layer = getattr(self, f'norm_layer{i + 1}')

            x, H, W = input_embed(x)

            for stg in stage:
                x = stg(x)

            x = norm_layer(x)
            outs.append(x)

        return outs



model = MSCANet(in_channnels=3, embed_dims=[32, 64, 460,256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3,3,5,2],
                 num_stages = 4, ls_init_val=1e-2, drop_path=0.0)
# summary(model, (3,1024,2048))

print(model)
y = torch.randn((6,3,1024,2048))#.to('cuda' if torch.cuda.is_available() else 'cpu')
x = model.forward(y)
print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (
        sum(p.numel() for p in model.parameters()) / 1000000.0))
for i in range(4):
    print(x[i].shape)


