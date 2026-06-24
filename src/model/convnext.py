import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from torch import nn
from torch import Tensor
from typing import List

from torchvision.ops import StochasticDepth



class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)),
                                  requires_grad=True)
    def forward(self, x):
        return self.gamma[None, ..., None, None] * x

class ConvNormAct(nn.Sequential):
    """
    A little util layer composed by (conv) -> (norm) -> (act) layers.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: int,
            norm=nn.BatchNorm2d,
            act=nn.ReLU,
            **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ),
            norm(out_features),
            act(),
        )


class BottleNeckBlock(nn.Module):
    #  就是之前的 Bottleneck
    def __init__(
            self,
            in_features: int,
            out_features: int,
            reduction: int = 4,
            stride: int = 1,
    ):
        super().__init__()
        reduced_features = out_features // reduction
        self.block = nn.Sequential(
            # wide -> narrow
            ConvNormAct(
                in_features, reduced_features, kernel_size=1, stride=stride, bias=False
            ),
            # narrow -> narrow
            ConvNormAct(reduced_features, reduced_features, kernel_size=3, bias=False),
            # narrow -> wide
            ConvNormAct(reduced_features, out_features, kernel_size=1, bias=False, act=nn.Identity),
        )
        self.shortcut = (
            nn.Sequential(
                ConvNormAct(
                    in_features, out_features, kernel_size=1, stride=stride, bias=False
                )
            )
            if in_features != out_features
            else nn.Identity()
        )

        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        res = self.shortcut(res)
        x += res
        x = self.act(x)
        return x

class ResnetStage(nn.Sequential):
    # 这个实际上就是 resnet 里面的 makelayer   这里是包含了很多个 BottleNeckBlock
    def __init__(
        self, in_features: int, out_features: int, depth: int, stride: int = 2, **kwargs
    ):
        super().__init__(
            # downsample is done here
            BottleNeckBlock(in_features, out_features, stride=stride, **kwargs),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth - 1)
            ],
        )
class ResnetStem(nn.Sequential):
    # 这个就是原始的 resnet 的 最前几个 卷积 BN relu和最大maxpool
    # 也就是 初始的
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            ConvNormAct(
                in_features, out_features, kernel_size=7, stride=2
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

class ResnetEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            stem_features: int,
            depths: List[int],
            widths: List[int],
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)
        in_out_widths = list(zip(widths, widths[1:])) # （64,128）（128,256）（256,512）
        self.stages = nn.ModuleList(
            [
                ResnetStage(stem_features, widths[0], depths[0], stride=1),
                *[
                    ResnetStage(in_features, out_features, depth)
                    for (in_features, out_features), depth in zip(
                        in_out_widths, depths[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x





#  resnet   --> convnext
#  depths=[3, 4, 6, 4]   -->  depths=[3, 3, 9, 3]
#  stem  从 7*7 +   最大池化层 --> 4*4 stride = 4


# ----------------------以下是手动实现的convnext -------------------------------------
#  stem  从 7*7 +   最大池化层 --> 4*4 stride = 4
class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_features)
        )

#  BottleNeck 中的 3x3 卷积层采用分组卷积来减少 FLOPS
#  宽 -> 窄 -> 宽 修改到到 窄 -> 宽 -> 窄
# 使用更大的内核尺寸(7x7)

# relu -> gelu
 # 减少 归一化 的次数   仅保留一个 第一层
 # 减少 激活的次数 ，之保留中间的
class ConvNextBottleNeckBlock(nn.Module):
    #  就是之前的 Bottleneck 改写的
    def __init__(
            self,
            in_features: int,
            out_features: int,
            expansion: int = 4,
            drop_p: float = .0,
            layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(
                in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x
class ConvNexStage(nn.Sequential):
    # 这个实际上就是 resnet 里面的 makelayer   这里是包含了很多个 BottleNeckBlock
    # 下采样 更换为 2* 2
    def __init__(
        self, in_features: int, out_features: int, depth: int, stride: int = 2, **kwargs
    ):
        super().__init__(
            # downsample is done here
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
            ),
            *[
                #ConvNextBottleNeckBlock(out_features, out_features, **kwargs)
                ConvNeXtBlock(out_features, **kwargs)
                for _ in range(depth - 1)
            ],
        )
class ClassificationHead(nn.Sequential):
    def __init__(self, num_channels: int, num_classes: int = 1000):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_classes)
        )
class ConvNextForImageClassification(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 stem_features: int,
                 depths: List[int],
                 widths: List[int],
                 drop_p: float = .0,
                 num_classes: int = 1000):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels, stem_features, depths, widths, drop_p)
        self.head = ClassificationHead(widths[-1], num_classes)
class ConvNextEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            stem_features: int,
            depths: List[int],
            widths: List[int],
            drop_p: float = .0,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_path=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_path=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


#      -------------下面是原声的--------------

from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class  ConvNeXtBlock(nn.Module):
    #原来的
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        # ========================= 下采样块 分别是 初始图像浅层采样的代码 、各个 stage 前面的下采样层  ==========================
        self.downsample_layers = nn.ModuleList()
        # ------------------------------------------ stem 是 初始的图像采样 ------------------------------------------------
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        # ------------------------------------------ 这下面 是 后续stage前面的图像降采样 ------------------------------------------------
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        # ------------------------------------------ 这下面 进行
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = LayerNorm(dims[0], eps=1e-6, data_format="channels_first") #partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = LayerNorm(dims[i_layer], eps=1e-6, data_format="channels_first")
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        # TUMPLe(out）
        return x_out

    def forward(self, x):
        x = self.forward_features(x)
        return x



class ConvNextForImageClassification2(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 stem_features: int,
                 depths: List[int],
                 widths: List[int],
                 drop_p: float = .0,
                 num_classes: int = 1000):
        super().__init__()

        self.encoder = ConvNeXt(in_chans=in_channels, depths=depths, dims=widths,
                 drop_path_rate=drop_p, layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3])
        self.head = ClassificationHead(widths[-1], num_classes)




def Convnext50():
    classifier = ConvNextForImageClassification(in_channels=3, stem_features=64, depths=[3, 3, 9, 3],
                                                widths=[96, 192, 384, 768],drop_p=0.5,num_classes=11)

    return classifier



def Convnext_T():
    classifier = ConvNextForImageClassification2(in_channels=3, stem_features=96, depths=[3, 3, 9, 3],
                                                widths=[96, 192, 384, 768],num_classes=11)

    return classifier

def make_model(args, parent=False):
    return  ConvNextForImageClassification2(in_channels=1, stem_features=96, depths=[3, 3, 9, 3],
                                                widths=[96, 192, 384, 768],num_classes=args.num_class)



if __name__ == '__main__':
    import torch

    image = torch.rand(11, 3, 224, 224)
    encoder = ConvNextEncoder(in_channels=3, stem_features=64, depths=[3, 3, 9, 3], widths=[96, 192, 384, 768])
    print(encoder(image).shape )

    print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (
            sum(p.numel() for p in encoder.parameters()) / 1000000.0))
    model = Convnext_T()
    print(model)
    print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (
            sum(p.numel() for p in model.parameters()) / 1000000.0))

    print(model(image).shape)

