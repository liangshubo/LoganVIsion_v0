
import torch
from networkx.utils import groups
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import torchvision
from .backbone import mobilenetv2, resnet

# ----------------------------------------------

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

       # >>> m = torchvision.models.resnet18(pretrained=True)
       # >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
       # >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
       # >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
       # >>> out = new_m(torch.rand(1, 3, 224, 224))
       # >>> print([(k, v.shape) for k, v in out.items()])
       # >>>     [('feat1', torch.Size([1, 64, 56, 56])),
       # >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        # return_layer 是一个字典 {层：名称 ，层2：名称}
        # set(return_layer) 是将字典的键提取变成了 集合

        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# -----------------------------------ASPP 相关 ----------------------------
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [  # 这个卷积就比正常卷积多了个dilation    padding和dilation一样为了确保输出特征图大小固定
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),   # 在K=3 的时候 Padding = Dilation  可以维持 输出的   特征图
            nn.BatchNorm2d(out_channels),
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
            nn.Dropout(0.3), )  # 这里居然有丢弃层 知道这个东西 从来没见过有人用

    def forward(self, x):
        res = []
        for conv in self.convs:
            # print(conv(x).shape)
            res.append(conv(x))
        res = torch.cat(res, dim=1)  # 将五个输出拼接在一起
        return self.project(res)  # 然后再做一个1*1卷积+bn+relu+丢弃层


# --------------------------------Head --------------------------
class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):  # 2048 -- 256 --
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        ###   channel
        self.plane_channel_increse = nn.Sequential(
            nn.Conv2d(num_classes, 256, 1,groups=8),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1024, 1,groups=256),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, in_channels, 1,groups=256)
        )



        self._init_weight()

    def forward(self, feature,channel):

        b, c = channel.shape
        channel = channel.view(b, c, 1, 1)
        #  channel process
        channel2 = channel.view(b, c, 1, 1).to(torch.float)
        weight_channel = self.plane_channel_increse(channel2)

        low_level_feature = self.project(feature['low_level'])  # 1*1的卷积+bn+relu

        output_feature = self.aspp(feature['out']*weight_channel)

        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:],
                                       mode='bilinear', align_corners=False)  # 做个上采样

        classifier = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))


        out = classifier * channel
        #print(out.shape, )
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

##-----------------------------Model --------------------------


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self,listinput ):
        [x, y] = listinput
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.classifier(features,y)

        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

#  -----------------------------加载 mobile net    -----------------------

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name == 'deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    #elif name == 'deeplabv3':
        #return_layers = {'high_level_features': 'out'}
        #classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

#  -----------------------------加载 resnet 残差网络  -----------------------

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]  # 空洞卷积隔的步数

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}  #
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    # elif name == 'deeplabv3':
    #    return_layers = {'layer4': 'out'}
    #    classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    # 提取网络的第几层输出结果并给一个别名
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)  # 返回

    model = DeepLabV3(backbone, classifier)
    return model


# -------------------加载模型 ---------------------

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone == 'mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride,
                                pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model

#--------------------- Deeplab v3+--------------------

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)

def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)

def make_model(args, parent=False):
    return  deeplabv3plus_resnet50(args.num_class,pretrained_backbone=False)   #


if __name__ == '__main__':
    import torch
    model = deeplabv3plus_resnet50(num_classes=24,pretrained_backbone=False)
    x = torch.randn([10, 2, 320, 320])
    y = torch.randn([10,24])
    print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (
            sum(p.numel() for p in model.parameters()) / 1000000.0))
    out = model([x,y])
    print(out.shape)
    print(model)
