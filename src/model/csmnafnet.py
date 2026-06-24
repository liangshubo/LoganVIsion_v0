from .csm import CSM
from .nafnet import NAFNet
import torch.nn as nn 
import torch

class csmnafnet(nn.Module):
    def __init__(self):
        super(csmnafnet, self).__init__()
        self.shallow_det = CSM()
        self.shallow_rem = NAFNet(2,1)
    def forward(self, x):
        shallow = self.shallow_det(x)
        input = torch.cat([x,shallow],dim=1)
        out = self.shallow_rem(input)
        return shallow,out



            
def make_model(args):
    #model =baselineUnet(1,Layer=BaseLayer,block=BaseBlock,kersize=3) # EX3   kersize =3 .EX4 kersize = 7 ;; 8EX2
    # model = baselineUnetRFDN(1,Layer=BaseLayer2,block1=BaseBlock,block2 = RFDBBlock,kersize=3) 
    
    #model =baselineUnet_light(1,Layer=BaseLayer,block=BaseBlock,kersize=3) # EX1，EX2,   8EX1,8EX2   # Light_Unet  1,Layer=BaseLayer,block=BaseBlock,kersize=3
    
    model = csmnafnet()
    return model

if __name__ == '__main__':
    import thop
    from thop import clever_format
    from thop import profile
    model = make_model(1).cuda()
    print(model)
    input = torch.randn([1, 1, 592, 720]).cuda()

    out = model(input)
    print(out[0].shape,out[1].shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops : {}'.format(flops))
    print('params : {}'.format(params))