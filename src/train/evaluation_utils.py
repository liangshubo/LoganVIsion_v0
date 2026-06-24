


import torch
import SimpleITK as sitk


def tensor2array(tensor,rgb_range):
    tensor = tensor.squeeze(0)
    array = tensor.squeeze(0).cpu().numpy()*(255/rgb_range)
    return array


def save_prediction(data, path,rgb_range):
    '''
    input tensor 
    '''
    data = data.detach().squeeze(0).clamp(0,1).cpu().numpy().astype('float')*(255/rgb_range)
    #print(data.max())
    img_sitk = sitk.GetImageFromArray(data)
    #print(data.max())
    sitk.WriteImage(img_sitk, path)
    
#save_prediction(out,"/ultrasound/LiangShubo/DenoiseCode/autodn/test_code/ABDOME0C_nii.nii")