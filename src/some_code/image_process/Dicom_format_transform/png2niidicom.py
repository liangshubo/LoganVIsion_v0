import cv2
import torch
import torch.nn as nn

import argparse

args = None
import SimpleITK as sitk


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=10):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out


def make_model(args, parent=False):
    return DnCNN(1)


image = "/ultrasound/LiangShubo/DenoiseCode/autodn/Code/dataset/denoise/denoise/ABDO/CV=0/ABDOME0C"

dicom_file = sitk.ReadImage(image)

image = sitk.GetArrayFromImage(dicom_file).squeeze()[:, :, 0]

model = make_model(args).cpu()

model.load_state_dict(torch.load(
    r'/ultrasound/LiangShubo/DenoiseCode/autodn/Code/experiment/[DnCNN]-[denoise]-[2023-08-18-07-11]/model/model_best.pt'))
image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)
print(model)
print(image)
out = model(image)
print(out.shape)
from torchvision.utils import save_image

save_image(out / 255, "/ultrasound/LiangShubo/DenoiseCode/autodn/test_code/testtoday_png.png")


def save_prediction(data, path):
    '''
    input tensor
    '''
    data = data.detach().squeeze(0).unsqueeze(3).cpu().numpy().astype('float')

    print(data.shape)
    img_sitk = sitk.GetImageFromArray(data)
    sitk.WriteImage(img_sitk, path)


save_prediction(out, "/ultrasound/LiangShubo/DenoiseCode/autodn/test_code/ABDOME0C_nii.nii")
save_prediction(out, "/ultrasound/LiangShubo/DenoiseCode/autodn/test_code/testtoday_dicom.dcm")