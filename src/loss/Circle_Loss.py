import torch
import torch.nn.functional as F
import numpy as np
from torch.xpu import device


def sobel_filters(device):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    ky = kx.T
    kx = kx.reshape(1,1,3,3)
    ky = ky.reshape(1,1,3,3)
    kx = torch.from_numpy(kx).to(device)
    ky = torch.from_numpy(ky).to(device)
    return kx, ky

def perimeter_loss(pred, device):
    # pred: [B,1,H,W] in [0,1]
    b,c,h,w = pred.shape
    kx, ky = sobel_filters(device)
    grad_mag = 0
    for i in range(1,c):
        single_pred = pred[:,i:i+1,:,:]
        gx = F.conv2d(single_pred, kx, padding=1)
        gy = F.conv2d(single_pred, ky, padding=1)
        grad_mag += torch.sqrt(gx*gx + gy*gy + 1e-8)

    return grad_mag.mean()

def tv_loss(pred):
    # pred: [B,1,H,W]
    b,c,h,w = pred.shape

    grad_mag = 0
    for i in range(1,c):
        dh = torch.abs(pred[:,:,1:,:] - pred[:,:,:-1,:]).mean()

        dw = torch.abs(pred[:,:,:,1:] - pred[:,:,:,:-1]).mean()

        grad_mag +=(dh + dw)

    return grad_mag



if __name__ == '__main__':

    x = torch.ones([1,11,212,212]).to("cuda")
    device = "cuda"
    y = tv_loss(x)
    print(y)