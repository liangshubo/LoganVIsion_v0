# -*- coding: UTF-8 -*-
"""
Create on 2023-7-18
@Author: LiangShubo
@email: liangshubo@neusoftmedical.com

"""
import SimpleITK as sitk
import os
import cv2
import glob

import numpy as np
from skimage.metrics import mean_squared_error #as mse
from skimage.metrics import peak_signal_noise_ratio #as psnr
from skimage.metrics import structural_similarity #as ssim

# 将测试数据集的标签与图像都裁剪出ROI图像
dataset_path = 'E:\Denoise\clearview_data\clearviewdataset'

def cala_light(img):
    avg_lisght = img.mean()
    return avg_lisght

def contrast(img1):
    #img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #彩色转为灰度图片
    m, n = img1.shape
    #图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0   # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 +
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)
    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式
    return cg

def cala_grad(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # 计算垂直方向梯度
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度的幅值和方向
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 计算平均梯度
    average_gradient = np.mean(gradient_magnitude)
    return average_gradient



def cala_enl(image):
    mean_intensity = np.mean(image)
    variance_intensity = np.var(image)
    # 计算等效视数
    equivalent_number_of_looks = mean_intensity ** 2 / variance_intensity
    return equivalent_number_of_looks




def calculate_mscn(dis_image):
    dis_image = dis_image.astype(np.float32)  # 类型转换十分重要
    ux = cv2.GaussianBlur(dis_image, (7, 7), 7/6)
    ux_sq = ux*ux
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(dis_image**2, (7, 7), 7/6)-ux_sq))

    mscn = (dis_image-ux)/(1+sigma)

    return mscn
# Function to segment block edges
def segmentEdge(blockEdge, nSegments, blockSize, windowSize):
    # Segment is defined as a collection of 6 contiguous pixels in a block edge
    segments = np.zeros((nSegments, windowSize))
    for i in range(nSegments):
        segments[i, :] = blockEdge[i:windowSize]
        if(windowSize <= (blockSize+1)):
            windowSize = windowSize+1

    return segments
def noticeDistCriterion(Block, nSegments, blockSize, windowSize, blockImpairedThreshold, N):
    # Top edge of block
    topEdge = Block[0, :]
    segTopEdge = segmentEdge(topEdge, nSegments, blockSize, windowSize)

    # Right side edge of block
    rightSideEdge = Block[:, N-1]
    rightSideEdge = np.transpose(rightSideEdge)
    segRightSideEdge = segmentEdge(
        rightSideEdge, nSegments, blockSize, windowSize)

    # Down side edge of block
    downSideEdge = Block[N-1, :]
    segDownSideEdge = segmentEdge(
        downSideEdge, nSegments, blockSize, windowSize)

    # Left side edge of block
    leftSideEdge = Block[:, 0]
    leftSideEdge = np.transpose(leftSideEdge)
    segLeftSideEdge = segmentEdge(
        leftSideEdge, nSegments, blockSize, windowSize)

    # Compute standard deviation of segments in left, right, top and down side edges of a block
    segTopEdge_stdDev = np.std(segTopEdge, axis=1)
    segRightSideEdge_stdDev = np.std(segRightSideEdge, axis=1)
    segDownSideEdge_stdDev = np.std(segDownSideEdge, axis=1)
    segLeftSideEdge_stdDev = np.std(segLeftSideEdge, axis=1)

    # Check for segment in block exhibits impairedness, if the standard deviation of the segment is less than blockImpairedThreshold.
    blockImpaired = 0
    for segIndex in range(segTopEdge.shape[0]):
        if((segTopEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segRightSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segDownSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segLeftSideEdge_stdDev[segIndex] < blockImpairedThreshold)):
            blockImpaired = 1
            break

    return blockImpaired
def noiseCriterion(Block, blockSize, blockVar):
    # Compute block standard deviation[h,w,c]=size(I)
    blockSigma = np.sqrt(blockVar)
    # Compute ratio of center and surround standard deviation
    cenSurDev = centerSurDev(Block, blockSize)
    # Relation between center-surround deviation and the block standard deviation
    blockBeta = (abs(blockSigma-cenSurDev))/(max(blockSigma, cenSurDev))

    return blockSigma, blockBeta

# Function to compute center surround Deviation of a block
def centerSurDev(Block, blockSize):
    # block center
    center1 = int((blockSize+1)/2)-1
    center2 = center1+1
    center = np.vstack((Block[:, center1], Block[:, center2]))
    # block surround
    Block = np.delete(Block, center1, axis=1)
    Block = np.delete(Block, center1, axis=1)

    # Compute standard deviation of block center and block surround
    center_std = np.std(center)
    surround_std = np.std(Block)

    # Ratio of center and surround standard deviation
    cenSurDev = (center_std/surround_std)

    # Check for nan's
    # if(isnan(cenSurDev)):
    #     cenSurDev = 0

    return cenSurDev
def cala_piqe(im):
    blockSize = 16  # Considered 16x16 block size for overall analysis
    activityThreshold = 0.1  # Threshold used to identify high spatially prominent blocks
    blockImpairedThreshold = 0.1  # Threshold identify blocks having noticeable artifacts
    windowSize = 6  # Considered segment size in a block edge.
    nSegments = blockSize-windowSize+1  # Number of segments for each block edge
    distBlockScores = 0  # Accumulation of distorted block scores
    NHSA = 0  # Number of high spatial active blocks.

    # pad if size is not divisible by blockSize
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    originalSize = im.shape
    rows, columns = originalSize
    rowsPad = rows % blockSize
    columnsPad = columns % blockSize
    isPadded = False
    if(rowsPad > 0 or columnsPad > 0):
        if rowsPad > 0:
            rowsPad = blockSize-rowsPad
        if columnsPad > 0:
            columnsPad = blockSize-columnsPad
        isPadded = True
        padSize = [rowsPad, columnsPad]
    im = np.pad(im, ((0, rowsPad), (0, columnsPad)), 'edge')

    # Normalize image to zero mean and ~unit std
    # used circularly-symmetric Gaussian weighting function sampled out
    # to 3 standard deviations.
    imnorm = calculate_mscn(im)

    # Preallocation for masks
    NoticeableArtifactsMask = np.zeros(imnorm.shape)
    NoiseMask = np.zeros(imnorm.shape)
    ActivityMask = np.zeros(imnorm.shape)

    # Start of block by block processing
    total_var = []
    total_bscore = []
    total_ndc = []
    total_nc = []

    BlockScores = []
    for i in np.arange(0, imnorm.shape[0]-1, blockSize):
        for j in np.arange(0, imnorm.shape[1]-1, blockSize):
             # Weights Initialization
            WNDC = 0
            WNC = 0

            # Compute block variance
            Block = imnorm[i:i+blockSize, j:j+blockSize]
            blockVar = np.var(Block)

            if(blockVar > activityThreshold):
                ActivityMask[i:i+blockSize, j:j+blockSize] = 1
                NHSA = NHSA+1

                # Analyze Block for noticeable artifacts
                blockImpaired = noticeDistCriterion(
                    Block, nSegments, blockSize-1, windowSize, blockImpairedThreshold, blockSize)

                if(blockImpaired):
                    WNDC = 1
                    NoticeableArtifactsMask[i:i +
                                            blockSize, j:j+blockSize] = blockVar

                # Analyze Block for guassian noise distortions
                [blockSigma, blockBeta] = noiseCriterion(
                    Block, blockSize-1, blockVar)

                if((blockSigma > 2*blockBeta)):
                    WNC = 1
                    NoiseMask[i:i+blockSize, j:j+blockSize] = blockVar

                # Pooling/ distortion assigment
                # distBlockScores = distBlockScores + \
                #     WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2)

                if WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2) > 0:
                    BlockScores.append(
                        WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2))

                total_var = [total_var, blockVar]
                total_bscore = [total_bscore, WNDC *
                                (1-blockVar) + WNC*(blockVar)]
                total_ndc = [total_ndc, WNDC]
                total_nc = [total_nc, WNC]

    BlockScores = sorted(BlockScores)
    lowSum = sum(BlockScores[:int(0.1*len(BlockScores))])
    Sum = sum(BlockScores)
    Scores = [(s*10*lowSum)/Sum for s in BlockScores]
    C = 1
    Score = ((sum(Scores) + C)/(C + NHSA))*100

    # if input image is padded then remove those portions from ActivityMask,
    # NoticeableArtifactsMask and NoiseMask and ensure that size of these masks
    # are always M-by-N.
    if(isPadded):
        NoticeableArtifactsMask = NoticeableArtifactsMask[0:originalSize[0],
                                                          0:originalSize[1]]
        NoiseMask = NoiseMask[0:originalSize[0], 0:originalSize[1]]
        ActivityMask = ActivityMask[0:originalSize[0], 1:originalSize[1]]

    return Score


INFORMATION = []

def resize_and_save(path,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    list  = os.listdir(path)
    count = 1
    for imgname in list:
        img = os.path.join(path,imgname)
        dicom = sitk.ReadImage(img)
        image = sitk.GetArrayFromImage(dicom).squeeze()[:,:,0]
        image = image[180:756,424:1192]
        #print(image.shape)
        cv2.imwrite(os.path.join(save_path,imgname+".png"),image,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print("finish dicom2png imwrite [{}/{}]".format(count,len(list)))
        count += 1

def read_image(path,name_list):
    img_array_list = []
    for name in name_list:
        img = os.path.join(path,name+".png")
        img_array = cv2.imread(img,0)
        img_array_list.append(img_array)
    return img_array_list

def cala_psnr_and_txt(img1,img2):
    '''
    :param img1:array
    :param img2:array
    :return: psnr and ssim
    '''
    #print(img1.shape,"img1")
    #print(img2.shape,"img2")
    psnr = peak_signal_noise_ratio(img1, img2)
    ssim = structural_similarity(img1, img2, multichannel=True)
    #print("PSNR {:.6f} SSIM {:.6f}".format(psnr, ssim))
    return psnr,ssim

def read_and_psnr_name(imgpath,name_list,name,label_array_list):
    img_array_list = read_image(imgpath,name_list) # 根据文件夹路径以及名字列表 就可以读取图像的数组列表
    AVG_PSNR = 0
    AVG_SSIM = 0
    PSNR_FILE = {}
    SSIM_FILE = {}
    for i in range(len(name_list)):
        psnr,ssim = cala_psnr_and_txt(img_array_list[i],label_array_list[i]) # 对应的图像对进行计算过程中
        PSNR_FILE[name_list[i]] = psnr  # 字典，将文件名以及对应的数值指标进行键值对
        SSIM_FILE[name_list[i]] = ssim
        #print(name_list[i],"PSNR {:.6f} SSIM {:.6f}".format(psnr, ssim))
        AVG_PSNR+=psnr
        AVG_SSIM+=ssim

        font = cv2.FONT_HERSHEY_SIMPLEX
        info1 = name
        cv2.putText(img_array_list[i],info1,(10,30),font,1,(255,255,255),2)
        # 指标信息
        info2 = "PSNR:{:.2f}".format(psnr)
        info3 = "SSIM:{:.2f}".format(ssim)
        cv2.putText(img_array_list[i],info2,(10,60),font,1,(255,255,255),2)
        cv2.putText(img_array_list[i], info3, (10, 90), font, 1, (255, 255, 255), 2)

    # 排序顺序
    SORT_PSNR_FILE = dict(sorted(PSNR_FILE.items(),key=lambda item:item[1],reverse=True))
    #SORT_SSIM_FILE = dict(sorted(SSIM_FILE.items(), key=lambda item: item[1], reverse=True))
    for filename,psnr in SORT_PSNR_FILE.items():
        print("FileName: [{:}] PSNR : [{:.4f}] SSIM: [{:.4f}]".format(filename,psnr,SSIM_FILE[filename]))
    # 输出平均的评价指标
    AVG_PSNR= AVG_PSNR/len(name_list)
    AVG_SSIM= AVG_SSIM/len(name_list)
    inro = imgpath+ " AVG_PSNR : {:.6f}  AVG_SSIM : {:.6f}".format(AVG_PSNR,AVG_SSIM)
    print(inro)
    INFORMATION.append(inro)
    return img_array_list





def calc_nriqa(image):
    light = cala_light(image)
    contrasts = contrast(image)
    grad = cala_grad(image)
    #piqe = cala_piqe(image)
    enl = cala_enl(image)
    return light,contrasts,grad,enl


def read_and_calc_iqa(imgpath,name_list,name):
    print("calcing {:} ....".format(name))
    img_array_list = read_image(imgpath, name_list)
    AVG_LIGHT = 0
    AVG_CONTRAST = 0
    AVG_GRAD = 0
    #AVG_PIQE = 0
    AVG_ENL = 0

    LIGHT_FILE = {}
    CONTRAST_FILE = {}
    GRAD_FILE = {}
    #PIQE_FILE = {}
    ENL_FILE = {}

    for i in range(len(name_list)):
        light,contrasts,grad,enl = calc_nriqa(img_array_list[i])
        LIGHT_FILE[name_list[i]] = light  # 字典，将文件名以及对应的数值指标进行键值对
        CONTRAST_FILE[name_list[i]] = contrasts
        GRAD_FILE[name_list[i]] = grad
        #PIQE_FILE[name_list[i]] = piqe
        ENL_FILE[name_list[i]] = enl

        AVG_LIGHT += light
        AVG_CONTRAST += contrasts
        AVG_GRAD += grad
        #AVG_PIQE += piqe
        AVG_ENL +=enl

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.7
        font_line = 2 
        
        info1 = name
        
        
        
        cv2.putText(img_array_list[i], info1, (10, 30), font, font_size, (255, 255, 255), font_line)
        # 指标信息
        info2 = "LIGHT:{:.2f}".format(light)
        info3 = "CONTRAST:{:.2f}".format(contrasts)
        info4 = "GRAD:{:.2f}".format(grad)
        #info5 = "PIQE:{:.2f}".format(piqe)
        info6 = "ENL:{:.2f}".format(enl)

        cv2.putText(img_array_list[i], info2, (10, 60), font, font_size, (255, 255, 255), font_line)
        cv2.putText(img_array_list[i], info3, (10, 90), font, font_size, (255, 255, 255), font_line)
        cv2.putText(img_array_list[i], info4, (10, 120), font, font_size, (255, 255, 255), font_line)
        #cv2.putText(img_array_list[i], info5, (10, 150), font, 1, (255, 255, 255), 2)
        cv2.putText(img_array_list[i], info6, (10, 150), font,font_size, (255, 255, 255), font_line)

    for filename,LIGHT in  LIGHT_FILE.items():
        print("FileName: [{:}] LIGHT: [{:.4f}]  CONTRAST:[{:.4f}]  GRAD:[{:.2f}]  ENL:[{:.2f}] ".format(filename,LIGHT_FILE[filename],CONTRAST_FILE[filename],GRAD_FILE[filename],ENL_FILE[filename]))

    AVG_LIGHT = AVG_LIGHT/ len(name_list)
    AVG_CONTRAST = AVG_CONTRAST/ len(name_list)
    AVG_GRAD =  AVG_GRAD/ len(name_list)
    #AVG_PIQE = AVG_PIQE/ len(name_list)
    AVG_ENL = AVG_ENL/ len(name_list)

    info = imgpath + " AVG_LIGHT: {:.4f}  AVG_CONTRAST:{:.4f}  AVG_GRAD:{:.4f} AVG_ENL:{:.4f}".format(AVG_LIGHT,AVG_CONTRAST,AVG_GRAD,AVG_ENL)
    print(info)
    INFORMATION.append(info)
    return img_array_list


def label_name(gt_array_list,name="CV=1"):
    for i in gt_array_list:
        font = cv2.FONT_HERSHEY_SIMPLEX
        info1 = name
        cv2.putText(i, info1, (10, 30), font, 1, (255, 255, 255), 2)
    return gt_array_list

def cat_array_list(array_list1_list):
    cat_image_array_list = []
    for array_idx in range(len(array_list1_list[0])):
        cat_array = array_list1_list[0][array_idx]
        for list_idx in range(1,len(array_list1_list)):
            cat_array = np.concatenate((cat_array,array_list1_list[list_idx][array_idx]),axis=1)
        cat_image_array_list.append(cat_array)
        del cat_array
    return cat_image_array_list

def save_cat_image(cat_image_array_list,name_list,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Save and Cat image ...")
    for i in range(len(cat_image_array_list)):
        cv2.imwrite(os.path.join(save_path,name_list[i]+".png"),cat_image_array_list[i],[cv2.IMWRITE_PNG_COMPRESSION,0])

def mutil_cat(raw_path,folder_list,image_name_list,label_name_listv1,label_name_listv3,label_name_listv5,save_path):
    path_list = [] # raw_path/folder_list[0] , raw_path/folder_list[1]  , raw_path/folder_list[2]
    for i in range(len(folder_list)):
        path = os.path.join(raw_path,folder_list[i])
        path_list.append(path)
    #label_path = path_list[-1] # 好像也不太需要什么label_path了

    array_list_list = []
    for i in range(len(path_list)):
        name = folder_list[i]  # folder_list[0]
        imagepath = path_list[i]  # raw_path/folder_list[2]
        if name.find("CV=1")>=0:
            array_list = read_and_calc_iqa(imagepath,label_name_listv1,name)
        elif name.find("CV=3")>=0:
            array_list = read_and_calc_iqa(imagepath,label_name_listv3,name)
        elif name.find("CV=5")>=0:
            array_list = read_and_calc_iqa(imagepath,label_name_listv5,name)
        else:
            array_list = read_and_calc_iqa(imagepath,image_name_list,name)

        array_list_list.append(array_list)
    # print(label_array_list)
    # label_array_list = label_name(label_array_list)
    # array_list_list.append(label_array_list)
    cat_image_array_list = cat_array_list(array_list_list)
    save_cat_image(cat_image_array_list,image_name_list,save_path)
    for  infor in INFORMATION:
        print(infor)

def txtname2list(path):
    image_path_list = []
    gt_path_list = []
    noise_name = []
    gt_name = []
    with open(path, 'r') as f:
        lines = f.readlines()
        # print(len(lines))
        for i in range(0, len(lines), 2):
            image_path_list.append(lines[i].rstrip())
            gt_path_list.append(lines[i + 1].rstrip())

    for i in range(len(image_path_list)):
        noise_path = image_path_list[i]
        gt_path = gt_path_list[i]
        noist_path_name = noise_path.split("CV")[1][3:]
        gt_path_name = gt_path.split("CV")[1][3:]
        noise_name.append(noist_path_name)
        gt_name.append(gt_path_name)
    return noise_name,gt_name








if __name__ == '__main__':
    # 上述几个文件夹的父文件夹
    raw_path = r"/ultrasound/LiangShubo/DenoiseCode/autodn/ABDEnhance/assessment"
    # 将noise 与 GT转换为png
    noise_path  = os.path.join(raw_path,"Noise") #E:\Denoise\ModelOutput\Compare_and_cat_N20ABDOPlus_CV1\Noise"
    gt_path = os.path.join(raw_path,"CV=1")  # E:\Denoise\ModelOutput\Compare_and_cat_N20ABDOPlus_CV1\CV=1"
    #resize_and_save(gt_path, gt_path)
    #resize_and_save(noise_path,noise_path)

    # 根据Noise-GT 测试图像的文件名字文档提取文件名对应的
    noise_gt_path_txt = os.path.join(raw_path,"N20ABDOCV1_V2.txt")  #r"E:\Denoise\ModelOutput\Compare_and_cat_N20ABDOPlus_CV1\N20ABDOcv1.txt"
    noise_gt_path_txtv3 = os.path.join(raw_path, "N20ABDOCV3_V2.txt")    
    noise_gt_path_txtv5 = os.path.join(raw_path, "N20ABDOCV5_V2.txt")   


    noise_name_list,label_name_listv1 = txtname2list(noise_gt_path_txt)
    _, label_name_listv3 = txtname2list(noise_gt_path_txtv3)
    _, label_name_listv5 = txtname2list(noise_gt_path_txtv5)

    # 几个文件夹的名字按顺序来ResBaseline-8EX7
    #folder_list = ["Noise","Baseline-8EX1","Baseline-8EX2","ResBaseline-8EX3","ResBaselinePer-8EX4","ResBaseline-8EX5","RealSRGAN-8EX6","ResBaseline-8EX7","ResBaseline-8EX7tune","CV=1","CV=3","CV=5"] # ,"RIDNet-EX2-1" ,"CBDNet-EX2-1" "CBDNet-EX2-1","RIDNet-EX2-1","CBDNet-4EX7" ,"CBDNet-2EX2","RIDNet-2EX2",
    #  "ResBaseline-8EX3", "ResBaselinePer-8EX4",   "Attention-TEX1","Attention-TEX3","Attention-TEX4","Attention-TEX5","Attention-TEX6"
    
    # "NAFNet-BlEX3","NAFNet-BlEX4PreLoss","NAFNet-BlEX5","NAFNet-BlEX7"
    # "NAFNet-SRBlEX1","NAFNet-SRBlEX2Perloss","RLFN-SREX1","RLFN-SREX2" 
    
    # "RLFN-SREX1","RLFN-SREX2","RLFN-SREX3"
    # "NTEX10+NBlEX12_150","NTEX10+NBlEX12_300","NTEX10+NBlEX12_250","NBlEX3+NTEX10"
    # "NAFNet-TEX10","NTEX10+NBL13_B2_E300","NTEX10+NBL13_B2_E15","Noise"
    # "Noise","NBlEX13-Bl8-E15","NBlEX13-Bl8-E60","NBlEX13-Bl8-E105","NBlEX13-Bl8-E300"
    folder_list = ["Noise","NTEX10+NBL13_B7_E150","NTEX10+NBL13_B4_E15","NTEX10+NBL13_B8_E15","NTEX10+NBL13_B4_E30","NTEX10+NBL13_B3_E60","NTEX10+NBL13_B3_E105"]
    # 保存的文件夹
    save_path =  os.path.join(raw_path,"Compare_Cat_Tblur")   #r"E:\Denoise\ModelOutput\Compare_and_cat_N20abdo_cv1\Compare_Cat"
    mutil_cat(raw_path,folder_list,noise_name_list,label_name_listv1,label_name_listv3,label_name_listv5,save_path)


    #resize_and_save(test_image_list,save_path = save_png)
