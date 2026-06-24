import os

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio #as psnr
from skimage.metrics import structural_similarity #as ssim

from DISTS_pytorch import DISTS
from numpy.fft import fft2, ifft2, fftshift
import numpy as np
import  cv2
def phase_corr_with_strength(img1, img2):
    f1 = img1 - img1.mean()
    f2 = img2 - img2.mean()
    F1, F2 = fft2(f1), fft2(f2)
    R = F1 * np.conj(F2)
    R /= np.abs(R) + 1e-8
    r = np.real(ifft2(R))
    r = fftshift(r)
    peak_val = np.max(r)
    peak_pos = np.unravel_index(np.argmax(r), r.shape)
    return peak_val, peak_pos

def shifted_ssim(img1, img2):
    _, peak_pos = phase_corr_with_strength(img1, img2)
    shift = np.array(peak_pos) - np.array(img1.shape)//2
    # 用subpixel shift对齐
    from scipy.ndimage import shift as imshift
    img2_aligned = imshift(img2, shift=-shift, order=3, mode='reflect')
    sim = ssim(img1, img2_aligned, data_range=img1.max()-img1.min())
    return sim

def perceptual_similarity(img1, img2, alpha=0.5):
    corr_peak, _ = phase_corr_with_strength(img1, img2)
    ssim_val = shifted_ssim(img1, img2)
    final_sim = alpha * corr_peak + (1 - alpha) * ssim_val
    return final_sim


import torch
import torch.nn as nn

from torchvision import models
import lpips

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.contentFunc = self.vgg()

    def vgg(self):
        conv_3_3_layer = 10
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):

            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def forward(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(torch.cat([fakeIm, fakeIm, fakeIm], dim=1))
        f_real = self.contentFunc.forward(torch.cat([realIm, realIm, realIm], dim=1))
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss




if __name__ == '__main__':



    path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/test/image/S4/S4LT_RXX0E"
    path2 = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset/rawdata/n20_all_dataset930/png/test/image/S3/S3MH_MSK04"
    namelist = os.listdir(path)
    namelist2 = os.listdir(path2)

    a = PerceptualLoss()
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    model = DISTS().cuda().eval()
    for i in range(len(namelist)-1):
        image1 = cv2.imread(os.path.join(path2,namelist2[i]),0)
        image2 = cv2.imread(os.path.join(path2, namelist2[i+1]),0)

        image1 = cv2.resize(image1, (512, 512), interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, (512, 512), interpolation=cv2.INTER_CUBIC)

        sisi = perceptual_similarity(image1, image2)

        ten1 = torch.tensor(image1 ).unsqueeze(0).float().cuda().unsqueeze(0)   # [-1,1]
        ten2 = torch.tensor(image2).unsqueeze(0).float().cuda().unsqueeze(0)
        perloss = a(ten1,ten2)
        print(i," - ",i+1,": ",sisi," -per: ", perloss.item(),"[lpips] :",lpips_fn(ten1,ten2).item(),"[dists] : ", model(ten1,ten2).item(), " [ssim ] : " , structural_similarity(image1, image2) , " psnr : ",peak_signal_noise_ratio(image1, image2))