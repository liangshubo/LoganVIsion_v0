import torch
import torch.nn as nn
import torch
from sympy.printing.octave import print_octave_code
from torch import nn


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
            modulation 就是调制 可变形卷积
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)

        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # 在这里就是 offset  的 获取
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)   # 这里面 的 代码需要注意  要保证 padding = kersize_size //2

        #
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        #print(x.shape,"rawdata.shape" )
        # [B,C,H,W]
        offset = self.p_conv(x)
        # 这个尺寸就是最后的输出尺寸  这个尺寸在conv_p 设置为 3,1,stride 时， 和外面的kersize  = 2p+1 输出一致
        #print(offset.shape,"offset.shape")


        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))   # 调制可变形卷积

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2   # N = kernel_size*kernel_size

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()    # 假设一个采样点的坐标为 (3.2 , 4.5) 则 lt 是 左上的坐标   (3,4)  rb 就是右下角的坐标位置  (4,5)   (b, h, w, 2N)
        q_rb = q_lt + 1  #  (b, h, w, 2N)

        #print(q_lt.shape , " q_lt ")
        #print(q_rb.shape , " q_rb ")

        # 前N 是行 ，所以截断最大数为 x.size(2) h , 后N 是列  ，所以截断在最大 x.size(3) w
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1) # 实际上 在这里的 rt 才是左下角

        # clip p

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)    计算权重  标注错误 但是 权重计算是 对的
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))

        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)     从上面的 采样坐标修正位置 上下左右 取出相应的 数据

        # 进行采样
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)    对应位置权重 * 对应位置的 数据 然后相加   索引 展平
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation     如果是有调制的 则加上一个权重的 乘法
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        #print(x_offset.shape , "  合成的   x_offset   尺寸  ")
        x_offset = self._reshape_x_offset(x_offset, ks)   # 这个尺寸是将所有的 经过偏移采样的 数据  进行变形
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        #

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # 中心点的物理坐标  ， 第一个卷积像素的索引 (0,0)  ，物理的坐标是 (0.5,0.5） ,
        # 这里 理论上应该是   O * stride   - padding  + floor(kersize)  +1  ，这里 padding 、stride 、kersize  都是在之前的卷积参数
        # 因此
        #
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        # print(p_0)
        # print(p_0.shape)   # [1,2n,h,w ]
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        # 这里是 获得当前核尺寸下的所有卷积位置相对于 中心点的 相对坐标 ，向下是H+ ； 向右是W+
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)

        # print(p_0.shape, "p_0.shape ")
        # print(p_n.shape, " p_n.shape")
        # print(offset.shape, " offset.shape")

        p = p_0 + p_n + offset   # offset = [1, 2N , h,w ]
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()  # B，H，W，2N
        padded_w = x.size(3)  # B，C，H，W
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)   # 这里也是 展平了

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y 前面是 行 后面是 列 ，展开的话，就是  行索引*宽度 + 列索引
        # 理论上一共有 N 个 位置 ，这里将行和列的 索引 合并
        # (b, c, h*w*N)
        #print(index.shape, " index  的尺寸  ") # [ 1,1,4,4,9 ] -> [1,c,1,4,4,9]  -> [ 1,c,4*4*9 ]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)   # N维度采样转换为 H，W
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)  #

        return x_offset



if __name__ == '__main__':
    layer = DeformConv2d(inc=1 ,outc=1,kernel_size=7,padding=3,stride=1 )
    layer2 = nn.Conv2d(1 ,1,kernel_size=7,padding=3,stride=3 )
    layer3 = nn.Conv2d(1, 1, kernel_size=3, padding=0, stride=3)
    x = torch.randn([1,1,10,10])
    print('\033[1;34m[ =======> Total params: %.2fM <======= ]\033[0m' % (
            sum(p.numel() for p in layer.parameters()) / 1000000.0))
    out = layer(x)
    print(out.shape,"last output shape ")
