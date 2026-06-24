import torch
import torch.nn as nn

from torchvision import models
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss,self).__init__()
        self.criterion = nn.MSELoss()
        self.contentFunc = self.vgg()
    def vgg(self):
        conv_3_3_layer = 10
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model
		
			
    def forward(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(torch.cat([fakeIm,fakeIm,fakeIm],dim=1))
        f_real = self.contentFunc.forward(torch.cat([realIm,realIm,realIm],dim=1))
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
    
    
a = PerceptualLoss()
