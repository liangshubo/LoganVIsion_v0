import cv2
import numpy as np
import imageio
import torch
import torch.utils.data as data
import math
#import SimpleITK as sitk
from abc import abstractmethod
from .base_dataset import BaseDataset   # from .base_dataset import BaseDataset
import os 
import random
import matplotlib
matplotlib.use('TkAgg')
# 图像分类数据集 输入txt ,其中每行都是path+" "+label
class Cv2ClassDataset(BaseDataset):
    def __init__(self,args,train_dataset_name=None):
        super(Cv2ClassDataset,self).__init__(args,train_dataset_name)


        self.dataset_image_pathfile = os.path.join(args.dataset_path,train_dataset_name,train_dataset_name+".txt")
        self.get_path_from_txt()
        self.resize_traindata = args.resize_traindata
        self.num_class = args.num_class
        
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

    def crop_image(self,img):
        h,w = img.shape
        (h1,h2,w1,w2) = (174,766,449,1169)
        if h2>h :
            h2 = h
        if w2>w:
            w2 = w 
        img = img[h1:h2,w1:w2]
        return img

    def __getitem__(self, idx):
        idx = idx % len(self.image_path_list)
        image = self.image_path_list[idx]
        gt = self.gt_path_list[idx]

        tensor_image,tensor_gt = self.load_file(image,gt)
        return tensor_image, tensor_gt

    def read_file(self,image): # gt 不需要 读取
        '''
        this define the image is one channel , if you want set rgb channel the imread(path) neednot (path,0)
        '''
        image = cv2.imread(image,0)

        return image
    
    def resize_image(self,image):
    
        def _resize(s):
            return cv2.resize(s,(self.resize_traindata,self.resize_traindata))
        ret = _resize(image)
        return ret

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

    @staticmethod
    def augment_image(image, rect_prob=0.6, noise_prob=0.6, blur_prob=0.6):
        """
        对单通道图像进行随机增强
        参数:
            image: 单通道灰度图像(二维numpy数组)
            rect_prob: 随机遮挡的触发概率(0.0~1.0)
            noise_prob: 高斯噪声的触发概率(0.0~1.0)
            blur_prob: 高斯模糊的触发概率(0.0~1.0)
        返回:
            增强后的图像(同尺寸单通道numpy数组)
        """
        # 确保是单通道图像
        assert len(image.shape) == 2, "Input must be a single-channel image"

        # 操作1: 随机遮挡(添加多个黑色方块)
        if random.random() < rect_prob:
            h, w = image.shape
            num_rects = random.randint(80, 250)  # 随机方块数量

            for _ in range(num_rects):
                # 随机方块尺寸(2-4像素)
                rect_size = random.randint(2, 8)

                # 随机位置(确保不越界)
                x = random.randint(0, w - rect_size - 1)
                y = random.randint(0, h - rect_size - 1)

                # 将选定区域设置为0(纯黑色)
                image[y:y + rect_size, x:x + rect_size] = random.choice([0,50,100,30,0,0])
        # 操作2: 随机添加高斯噪声
        if random.random() < noise_prob:
            # 生成与原图相同形状的噪声
            mean = random.randint(-2, 2)
            std = random.randint(0, 20)
            noise = np.random.normal(mean, std, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # 操作3: 随机添加高斯模糊
        if random.random() < blur_prob:
            # 随机选择奇数大小的卷积核(3,5,7)
            ksize = random.choice([3, 5, 7])
            # 随机标准差(0.5~2.0)
            sigma = random.uniform(0.5, 2.0)
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)

        return image

    @staticmethod
    def augement_rota_translate(image):

        def _rotate_augment_cover(img: np.ndarray,
                                 interp=cv2.INTER_LINEAR
                                 ):
            """
            旋转增广（保证覆盖原始位置，最终中心裁剪回原尺寸）。

            参数
            ----
            img : np.ndarray
                输入图像，需为正方形(H==W)，支持灰度(H,W)或彩色(H,W,C)。
            angle_deg : float
                旋转角度，期望在 [0, 45]（函数内部会clip到该范围）。
            interp : int
                OpenCV插值方式（默认双线性）。
            border_mode : int
                旋转时的边界填充模式（默认反射，避免黑边）。

            返回
            ----
            out_img : np.ndarray
                旋转+中心裁剪后的图像，尺寸与输入一致。
            info : dict
                计算信息，包括：
                - 'bbox_dim': 旋转后正方形的最小轴对齐包围框边长（int）
                - 'aabb': {'min_x','max_x','min_y','max_y','width','height'}  # 以缩放后坐标系为基准（左上为(0,0)）
                - 'extreme_points': {'top','bottom','left','right'} 四个极值点坐标 (x,y)（缩放后坐标系）
                - 'angle_deg': 实际使用的角度
            """

            angle_deg = random.randint(-10,10)

            # -------- 基本检查 --------
            if img.ndim not in (2, 3):
                raise ValueError("img 必须是灰度(H,W)或彩色(H,W,C)的numpy数组")
            H, W = img.shape[:2]
            if H != W:
                raise ValueError(f"输入需要正方形图像，但得到 H={H}, W={W}")
            N = H

            # -------- 角度与弧度 --------
            angle = float(np.clip(angle_deg, -40.0, 45.0))
            rad = math.radians(angle)

            # -------- 旋转后轴对齐包围框尺寸（正方形）--------
            # 对边长为 N 的正方形，以中心旋转 angle，其AABB宽/高均为 N*(|cos| + |sin|)
            c, s = abs(math.cos(rad)), abs(math.sin(rad))
            bbox_float = N * (c + s)
            bbox_dim = int(math.ceil(bbox_float))

            # -------- 1 \ 将原图  gt   缩放到 bbox_dim × bbox_dim --------
            if bbox_dim != N:
                resized = cv2.resize(img, (bbox_dim, bbox_dim), interpolation=interp)

            else:
                resized = img.copy()

            # -------- 2 \ 以中心旋转 resized --------
            center = (bbox_dim * 0.5, bbox_dim * 0.5)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            rotated_full = cv2.warpAffine(
                resized, M, (bbox_dim, bbox_dim),
                flags=interp,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )



            H1, W1 = rotated_full.shape[:2]
            out_img = rotated_full[int((H1 - H) / 2):int((H1 + H) / 2), int((W1 - W) / 2):int((W1 + W) / 2)]

            return out_img

        def _random_translate(img: np.ndarray,  max_shift: float,
                             interp=cv2.INTER_LINEAR):
            """
            随机方向平移数据增广

            参数
            ----
            img : np.ndarray
                输入图像，支持灰度(H,W)或彩色(H,W,C)。
            max_shift : float
                最大位移像素数（实际位移随机在 [0, max_shift]）。
            interp : int
                插值方式，默认双线性。

            返回
            ----
            out_img : np.ndarray
                平移后的图像，超出区域设为0。
            info : dict
                平移参数，包括 angle_deg, shift, dx, dy。
            """
            H, W = img.shape[:2]

            # 随机角度 (0-360度)
            angle = random.uniform(-180, 45)
            rad = math.radians(angle)

            # 随机位移大小
            shift = random.uniform(0, max_shift)
            dx = shift * math.cos(rad)
            dy = shift * math.sin(rad)

            # 仿射矩阵
            M = np.float32([[1, 0, dx],
                            [0, 1, dy]])

            # warpAffine 默认 borderValue=0 就是补零
            out_img = cv2.warpAffine(img, M, (W, H),
                                     flags=interp,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)



            return out_img
        if random.random()<0.5:

            rotate_image  = _rotate_augment_cover(image)

            translate_image  = _random_translate(rotate_image, max_shift=80)
        else:
            translate_image = image
        return  translate_image



    def augment_patch(self, image, hflip=True, rot=True):

        # if the patch_size = None the augment will be error because the image has filp and image 2 no flip the batch will be wrong
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if self.light_change == 1 and random.random() < 0.6:
            light_gain = [-20,-10,0, 10, 20, 30]
            light = random.choice(light_gain)
        else:
            light = 0

        def _augment(img):
            if hflip: img = img[:, ::-1]
            #if vflip: img = img[::-1, :]
            #if rot90: img = img.transpose(1, 0)
            if light != 0:
                imgf = img.astype(float)  # numpy  20   np.float -> float
                imgf_gain = imgf + light
                imgf_gain = np.clip(imgf_gain,0,255)
                img = imgf_gain# .astype(int)  # numpy  20   np.int -> int

            return img

        return _augment(image)

    def load_file(self, image,gt):
        '''
        image 是 一个路径
        gt 是一个数
        '''
        image = self.read_file(image)
        # print(image.shape,gt.shape)
        # 返回一个numpy数组，0-255 uint8 类型
        if self.crop_traindata:
            image = self.crop_image(image)

        if self.patch_size is not None:
            image= self.padding(image)
            image= self.get_patch(image)
            if self.argument_scale != 1:
                image = self.augment_patch(image)

        if self.resize_traindata is not None:
            image = self.resize_image(image)
            if self.argument_scale != 1:
                image = self.augment_patch(image)
                image = self.augment_image( image)   # 额外的数据增广、去噪、模糊、随机遮挡
                image= self.augement_rota_translate(image)

        # print(image.shape,gt.shape)
        image_tensor = self.np2tensor(image)

        #label_tensor = torch.zeros(self.num_class).float()
        #label_tensor[int(gt)] = 1.0
        label_tensor= torch.tensor(int(gt))
        return image_tensor, label_tensor


if __name__ == '__main__':
    import random
    from torchvision.transforms.functional import to_pil_image
    import matplotlib.pyplot as plt
    label_name  = ["S0","S1","S2","S3","S4","S5","S6","S7","S8","S10","S11"]
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
            image_tensor, label_tensor = dataset[idx]

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

    args.argument_scale = 8
    args.resize_traindata = 512
    args.light_argument = 1
    args.dataset_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/PreExp/dataset"
    dataset = Cv2ClassDataset(args,train_dataset_name="n20n6000_shoulder_class_all_1013")
    image,label = dataset.__getitem__(11)

    print(image)
    print(label)
    visualize_random_samples(dataset, num_samples=6)