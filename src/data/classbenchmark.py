import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data

#import SimpleITK as sitk
from abc import abstractmethod
from .base_dataset import BaseDataset   # from .base_dataset import BaseDataset
import os 
import random

# 图像分类数据集 输入txt ,其中每行都是path+" "+label
class ClassBenchmark(BaseDataset):
    def __init__(self,args,test_dataset_name=None):
        super(ClassBenchmark,self).__init__(args,test_dataset_name)

        # 这个是不一样的
        self.dataset_image_pathfile = os.path.join(args.dataset_path,'benchmark',test_dataset_name,test_dataset_name+".txt")
        self.get_path_from_txt()
        self.resize_traindata = args.resize_traindata
        self.num_class = args.num_class
        # 这是新增的属性
        self.test_dataset_name = test_dataset_name
    ### 一样的
    def get_path_from_txt(self):
        """
        the image and gt filepoath will read from self.dataset_image_pathfile
        and save in self.image_path_list\self.gt_path_list
        """
        with open(self.dataset_image_pathfile,'r') as f:
            lines = f.readlines()
            for i in range(0,len(lines)):

                self.image_path_list.append(lines[i].split(" ")[0])
                self.gt_path_list.append(lines[i].split(" ")[1])
    ### 一样的
    def crop_image(self,img):
        h,w = img.shape
        (h1,h2,w1,w2) = (174,766,449,1169)
        if h2>h :
            h2 = h
        if w2>w:
            w2 = w
        img = img[h1:h2,w1:w2]
        return img
    ### 有改的 取消增广
    def __len__(self):
        return len(self.image_path_list)

    ### 有改的
    def __getitem__(self, idx):
        idx = idx % len(self.image_path_list)
        image = self.image_path_list[idx]
        gt = self.gt_path_list[idx]

        tensor_image,tensor_gt,nameext  = self.load_file(image,gt)
        return tensor_image, tensor_gt,nameext

    ### 一样的
    def read_file(self,image): # gt 不需要 读取
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image = cv2.imread(image,0)

        return image
    ### 应该保证一样的
    def resize_image(self, image):

        def _resize(s):
            return cv2.resize(s, (self.resize_traindata, self.resize_traindata))

        ret = _resize(image)
        return ret
    ### 一样的
    def padding(self,image):
        ih, iw = image.shape[:2]
        hp = self.patch_size
        wp = self.patch_size

        def _padding(s):
            return cv2.copyMakeBorder(s, max(0, int(hp - ih)), max(0, int(hp - ih)), max(0, int(wp - iw)),
                                      max(0, int(wp - iw)), cv2.BORDER_CONSTANT, (0, 0, 0))

        if ih < hp or iw < wp:
            ret = [_padding(image)]
        else:
            ret = [image]

        return ret
    ### 一样的
    def augment_patch(self, image, hflip=True, rot=True):

        # if the patch_size = None the augment will be error because the image has filp and image 2 no flip the batch will be wrong
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        if self.light_change == 1 and random.random() < 0.5:
            light_gain = [-55, -45, -20, -10, 10, 20, 45, 55]
            light = random.choice(light_gain)
        else:
            light = 0

        def _augment(img):
            if hflip: img = img[:, ::-1]
            if vflip: img = img[::-1, :]
            if rot90: img = img.transpose(1, 0)
            if light != 0:
                imgf = img.astype(float)  # numpy  20   np.float -> float
                imgf_gain = imgf + light

                imgf_gain_high = (imgf_gain > 254)

                imgf_gain[imgf_gain_high] = 255
                img = imgf_gain.astype(int)  # numpy  20   np.int -> int
            return img

        return _augment(image)
    ### 改动
    def load_file(self, image,gt):
        '''
        image 是 一个路径
        gt 是一个数
        '''
        path, nameext = os.path.split(image)
        image = self.read_file(image)

        if self.crop_traindata:
            image = self.crop_image(image)

        if self.resize_traindata is not None:
            image = self.resize_image(image)

        image_tensor = self.np2tensor(image)
        #label_tensor = torch.zeros(self.num_class).float()
        #label_tensor[int(gt)] = 1.0
        label_tensor = torch.tensor(int(gt))
        return image_tensor, label_tensor,nameext


if __name__ == '__main__':
    import random
    from torchvision.transforms.functional import to_pil_image
    import matplotlib.pyplot as plt
    label_name  = ["S0","S1","S2","S3","S4","S6","S7","S8","S10","S11"]
    def visualize_random_samples(dataset, num_samples=5, figsize=(15, 10)):
        """
        可视化数据集中随机样本的图像和标签

        参数:
            dataset: 自定义数据集对象
            num_samples: 要显示的样本数量
            figsize: 图像显示大小
        """
        # 随机选择样本索引
        indices = random.sample(range(len(dataset)), num_samples)

        # 创建子图
        fig, axes = plt.subplots(1, num_samples, figsize=figsize)
        if num_samples == 1:
            axes = [axes]  # 确保单个样本时也能正确处理

        for idx, ax in zip(indices, axes):
            # 获取样本
            image_tensor, label_tensor,_= dataset[idx]
            image_tensor=image_tensor.squeeze(0)
            # 转换图像为可显示格式
            image = to_pil_image(image_tensor) if isinstance(image_tensor, torch.Tensor) else image_tensor

            # 获取标签类别
            label = torch.argmax(label_tensor).item() if label_tensor.dim() > 0 else label_tensor.item()
            label_vector = label_tensor.numpy() if isinstance(label_tensor, torch.Tensor) else label_tensor

            # 显示图像
            ax.imshow(image,cmap='gray')
            ax.axis('off')

            # 添加标题显示标签信息
            title = f"Label: {label_name[label]}\nOne-hot: {label_vector}"
            ax.set_title(title, fontsize=10)

        plt.tight_layout()
        plt.show()
    from src.option import  args
    args.resize_traindata = 512
    args.dataset_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset"
    dataset = ClassBenchmark(args,test_dataset_name="n6000_shoulder_class")
    image,label,nameext = dataset.__getitem__(0)

    print(image.size())
    print(label.size())
    visualize_random_samples(dataset, num_samples=6)

