import torch
import torch.nn as nn

from torchvision import models
import ssl
from loss.perceptualloss import PerceptualLoss


class GenerateLoss(nn.Module):
    def __init__(self) -> None:
        super(GenerateLoss,self).__init__()
        self.perception_loss = PerceptualLoss()
        self.L1 = nn.L1Loss()
    def forward(self,out_images,target_image):
        # Adaversarial_loss 
        
        perception_loss = self.perception_loss(out_images,target_image)
        image_loss = self.L1(out_images,target_image)
        
        return perception_loss+image_loss