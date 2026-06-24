import cv2

import numpy as np

import os 

import random

def image_cat_padding(name1,name2,name3,name4,noise_path,label_path):
    noise1 = np.load(os.path.join(noise_path,name1))
    noise2 = np.load(os.path.join(noise_path,name2))
    noise3 = np.load(os.path.join(noise_path,name3))
    noise4 = np.load(os.path.join(noise_path,name4))
    
    label1 = np.load(os.path.join(label_path,name1))
    label2 = np.load(os.path.join(label_path,name2))
    label3 = np.load(os.path.join(label_path,name3))
    label4 = np.load(os.path.join(label_path,name4))
    

    noise = [noise1,noise2,noise3,noise4]
    label = [label1,label2,label3,label4]
    
    (h1,w1) = noise1.shape
    (h2,w2) = noise2.shape
    (h3,w3) = noise3.shape
    (h4,w4) = noise4.shape
    
    H_list = [h1,h2,h3,h4]
    W_list = [w1,w2,w3,w4]
    
    noise_padding_cat = []
    label_padding_cat = []
    for i in range(len(H_list)):
        noise_array = noise[i]
        label_array = label[i]
        h,w = H_list[i],W_list[i]
        if h > 512:
            noise_array = noise_array[:512,:]
            label_array = label_array[:512,:]
        if w > 512:
            noise_array = noise_array[:,:512]
            label_array = label_array[:,:512]
        if h < 512:
            noise_array = cv2.copyMakeBorder(noise_array,int(np.floor((512-h)/2)),int(np.ceil((512-h)/2)),0,0,cv2.BORDER_CONSTANT, value=[0, 0, 0])
            label_array = cv2.copyMakeBorder(label_array,int(np.floor((512-h)/2)),int(np.ceil((512-h)/2)),0,0,cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if w < 512:
            noise_array = cv2.copyMakeBorder(noise_array,0,0,int(np.floor((512-w)/2)),int(np.ceil((512-w)/2)),cv2.BORDER_CONSTANT, value=[0, 0, 0])
            label_array = cv2.copyMakeBorder(label_array,0,0,int(np.floor((512-w)/2)),int(np.ceil((512-w)/2)),cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        hh,ww = noise_array.shape
    
        if hh ==512 and ww==512:
            
            noise_padding_cat.append(noise_array)
            label_padding_cat.append(label_array)
    
    [noise11,noise12,noise13,noise14] = noise_padding_cat
    [label11,label12,label13,label14] = label_padding_cat
    
    noise_1_3 = np.concatenate([noise11,noise13],axis=1)
    label_1_3 = np.concatenate([label11,label13],axis=1)
    noise_2_4 = np.concatenate([noise12,noise14],axis=1)
    label_2_4 = np.concatenate([label12,label14],axis=1)
    
    
    noise_1234 = np.concatenate([noise_1_3,noise_2_4],axis=0)
    label_1234 = np.concatenate([label_1_3,label_2_4],axis=0)
    
    return noise_1234,label_1234


def padding_and_cat(noise_path,label_path,save_noise=None,save_label=None):
    path_nameext = os.listdir(noise_path)
    num_list = [ i for i in range(len(path_nameext))]
    
    random.shuffle(num_list)
    #print(len(num_list))
    for i in range(0,len(num_list)-4,4):
        name1 = path_nameext[i]
        name2 = path_nameext[i+1]
        name3 = path_nameext[i+2]
        name4 = path_nameext[i+3]
        #print(name1,name2,name3,name4)        
        noise_1234,label_1234 =  image_cat_padding(name1,name2,name3,name4,noise_path,label_path)
        name = str(num_list[i])+"_"+str(num_list[i+1])+"_"+str(num_list[i+2])+"_"+str(num_list[i+3])
        np.save(os.path.join(save_noise,name+".npy"),noise_1234)
        np.save(os.path.join(save_label,name+".npy"),label_1234)
        print(f"finish save {i}/{len(num_list)}")
        
        
        
        
noise_path = r"/ultrasound/LiangShubo/DenoiseCode/Self-developed-SRI/dataset/rawdata/HeChengDataset403/noise2"
label_path = r"/ultrasound/LiangShubo/DenoiseCode/Self-developed-SRI/dataset/rawdata/HeChengDataset403/label2"
save_noise = r"/ultrasound/LiangShubo/DenoiseCode/Self-developed-SRI/dataset/rawdata/HeChengDataset_catpadding/noise2/"
save_label = r"/ultrasound/LiangShubo/DenoiseCode/Self-developed-SRI/dataset/rawdata/HeChengDataset_catpadding/label2/"
if not os.path.exists(save_noise):
    os.makedirs(save_noise)
    
if not os.path.exists(save_label):
    os.makedirs(save_label)
    

padding_and_cat(noise_path,label_path,save_noise,save_label)