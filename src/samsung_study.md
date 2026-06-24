### ------------------------------神经分割   USNS -----------------------------------------------

### ------------------------------ 神经 分割   SEGXNET      0.736   ----------------------------
python main_semantic_segment.py --model segxnet --resume -2 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 50 --batch_size 8  --patch_size 420   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path SegXnet-EX1 --num_class 2
python main_semantic_segment.py --model segxnet --resume 0 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 50 --batch_size 8  --patch_size 420   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [SegXnet-EX1]-[Ultrasound_Nerve_Segmention_Kaggle]-[2025-11-04-08-55] --num_class 2  --test_only 

python main_semantic_segment.py --model segxnet --resume -2 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou+1*FocalLoss --lr 1e-4  --epoch 30 --batch_size 8  --patch_size 420   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path SegXnet-EX1 --num_class 2**

#  Large    0.732  没有太多改进的  
python main_semantic_segment.py --model segxnet --resume -2 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou+1*FocalLoss --lr 1e-4  --epoch 100 --batch_size 14  --patch_size 420   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path SegXnet-Large-EX3 --num_class 2

##  


### ------------------------------ 神经 分割   CSAFNET       0.736   ----------------------------

### 神经 分割    0.736    USNS   
python main_semantic_segment.py --model csafnet_da --resume -2 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 50 --batch_size 8  --patch_size 420   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Csafnet-EX1 --num_class 2
python main_semantic_segment.py --model csafnet_da --resume 0 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 50 --batch_size 8  --patch_size 420   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [Csafnet-EX1]-[Ultrasound_Nerve_Segmention_Kaggle]-[2025-11-06-09-00] --num_class 2  --test_only

### ------------------------------ swinunet  -------------------------------------------------


### ----------------------------------神经 分割    SWINUNET  USNS  resize 448  直接训练的-------------------------------
# 训练   
python main_semantic_segment.py --model swinunet --resume -2 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 50 --batch_size 8  --resize_traindata 448    --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Swinunet-EX1 --num_class 2
# 测试 
python main_semantic_segment.py --model swinunet --resume 0 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 50 --batch_size 8  --resize_traindata 448    --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train  [Swinunet-EX1]-[Ultrasound_Nerve_Segmention_Kaggle]-[2025-11-13-11-09] --num_class 2 --test_only

# pretrain  
python main_semantic_segment.py --model swinunet --resume -2 --data_train US43d --data_test US43d --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 150 --batch_size 32  --resize_traindata 224    --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Swinunet-EX1-Pretrain --num_class 2
# fintune  (mIoU): 0.7324
python main_semantic_segment.py --model swinunet --resume 0 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 150 --batch_size 32  --resize_traindata 224   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [Swinunet-EX1-Pretrain]-[US43d]-[2025-11-14-13-00] --num_class 2  --ressave_path Swinunet-EX2-finetune
# test   Nerve   Class  Accuracy : 0.6936     Class Iou :0.4742     Class F1 :0.6433     Class Precision :0.5998    Class Recall :  0.6936    Class Dice : 0.6433         
python main_semantic_segment.py --model swinunet --resume 0 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 150 --batch_size 32  --resize_traindata 224   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [Swinunet-EX2-finetune]-[Ultrasound_Nerve_Segmention_Kaggle]-[2025-11-14-16-32]   --num_class 2  --test_only

###  - ---------------------------------deeplabv3_plus   ----------------------------------------------

#  这里是 0.739 
python main_semantic_segment.py --model deeplabv3 --resume -2 --data_train Ultrasound_Nerve_Segmention_Ka**g**gle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 150 --batch_size 32  --resize_traindata 448    --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Deeplab-EX1-Pretrain --num_class 2

python main_semantic_segment.py --model deeplabv3 --resume 0 --data_train Ultrasound_Nerve_Segmention_Kaggle --data_test Ultrasound_Nerve_Segmention_Kaggle --loss 1*CrossEntropy+1*Softmiou --lr 1e-4  --epoch 150 --batch_size 32  --resize_traindata 448    --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [Deeplab-EX1]-[Ultrasound_Nerve_Segmention_Kaggle]-[2025-11-19-09-21] --num_class 2  -test_only

# 预训练  US43d 

python main_semantic_segment.py --model deeplabv3 --resume -2 --data_train US43d --data_test US43d --loss 1*CrossEntropy+1*Dice+1*FocalLoss --lr 1e-4  --epoch 150 --batch_size 32  --resize_traindata 448    --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Deeplab-EX3-Pretrain --num_class 2


## 训练 N60000    +PRETRAIN    效果并不好的   
python main_semantic_segment.py --model deeplabv3 --resume 0 --data_train N6000_1125 --data_test N6000_1125 --loss 1*CrossEntropy+1*Dice+1*FocalLoss --lr 1e-4  --epoch 150 --batch_size 32  --patch_size 448    --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [Deeplab-EX1-Pretrain]-[US43d]-[2025-11-24-08-59] --num_class 2 

python main_semantic_segment.py --model deeplabv3 --resume 0 --data_train N6000_1125 --data_test N6000_1125 --loss 1*CrossEntropy+1*Dice+1*FocalLoss --lr 1e-4  --epoch 150 --batch_size 32  --patch_size 448    --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [Baseline-Thyroid_plus-Research-sam2]-[N6000_1125]-[2025-11-26-18-30] --num_class 2 --test_only

#  训练 N60000    +PRETRAIN     尝试 两个数据集一起训练 以及 使用resize     
python main_semantic_segment.py --model deeplabv3 --resume 0 --data_train N6000_1125+UBPB_Single --data_test N6000_1125 --loss 1*CrossEntropy+1*Dice+1*FocalLoss --lr 1e-4  --epoch 150 --batch_size 16  --resize_traindata 448     --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [Deeplab-EX1-Pretrain]-[US43d]-[2025-11-24-08-59] --ressave_path Deeplab-EX1-Finetune  --num_class 2 

python main_semantic_segment.py --model deeplabv3 --resume 0 --data_train N6000_1128+UBPB_Single --data_test N6000_1128 --loss 1*CrossEntropy+1*Dice+1*FocalLoss --lr 1e-4  --epoch 150 --batch_size 16  --resize_traindata 448     --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [Deeplab-EX1-Pretrain]-[US43d]-[2025-11-24-08-

 ====================================================================
  ====================================================================
  ====================================================================
  ====================================================================
 ====================================================================
  


## SR 流程测试     
### 训练 
python main_denoise.py --model rcan --resume -2 --data_train DIV2K4xto2x --data_test DIV2K4xto2x --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 8  --patch_size 224   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path RCAN-EX1 
### 测试 

python main_denoise.py --model rcan --resume 0 --data_train DIV2K4xto2x --data_test DIV2K4xto2x --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 8  --patch_size 224   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [RCAN-EX1]-[DIV2K4xto2x]-[2026-05-21-08-44] --scale 2 --test_only 

## 分割 流程测试 
### 训练 
python main_semantic_segment.py --model baselineunet_light --resume -2 --data_train USNS --data_test USNS --loss 1*CrossEntropy --lr 1e-4  --epoch 150 --batch_size 32  --resize_traindata 224    --print_every 100  --argument_scale 1  --light_argument 1  --rgb_range 1  --ressave_path BaseLine-EX1 --num_class 2

 ========================
 
## DIV2K   4倍率 训练  降采样用双三次    
### 训练
python main_denoise.py --model rcan --resume -2 --data_train DIV2K4x --data_test DIV2K4x --loss 1*L1+1*Percept --lr 1e-4  --epoch 100 --batch_size 4  --patch_size 128   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path RCAN-EX1 --scale 4
python main_denoise.py --model rcan --resume 0 --data_train set5 --data_test set5+DIV2K4x  --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 8  --patch_size 224   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [RCAN-EX1]-[DIV2K4x]-[2026-05-26-13-08] --scale 4 --test_only 


 ### ct0 X4  

#### 加载预训练模型  进行训练 
python main_denoise.py --model rcan --resume 0 --data_train CT0_AREA_X4 --data_test CT0_AREA_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 4  --patch_size 128   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [RCAN-EX1]-[DIV2K4x]-[2026-05-26-13-08] --scale 4

#### 继续训练     
python main_denoise.py --model rcan --resume -1 --data_train CT0_AREA_X4 --data_test CT0_AREA_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 4  --patch_size 128   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path  [TEST-EX1]-[CT0_AREA_X4]-[2026-05-28-12-55] --scale 4
#### 测试   
python main_denoise.py --model rcan --resume 0 --data_train CT0_AREA_X4 --data_test CT0_AREA_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 4  --patch_size 128   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train  [TEST-EX1]-[CT0_AREA_X4]-[2026-05-28-12-55] --scale 4  --test_only 



####   ct0   blur 模糊 加上  area  X4    

##### 训练
python main_denoise.py --model rcan --resume 0 --data_train CT0_BLUR_K5S1_AREA_X4 --data_test CT0_BLUR_K5S1_AREA_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 16   --patch_size 192   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [RCAN-EX1]-[DIV2K4x]-[2026-05-26-13-08] --scale 4  --ressave_path RCAN-EX2 

#### 测试
python main_denoise.py --model rcan --resume 0 --data_train CT0_BLUR_K5S1_AREA_X4 --data_test CT0_BLUR_K5S1_AREA_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 16   --patch_size 192   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --pre_train [RCAN-EX2]-[CT0_BLUR_K5S1_AREA_X4]-[2026-06-01-14-54] --scale 4  --test_only 


### rcan_Large    used DIV2K to Pretrain 


python main_denoise.py --model rcan_l --resume -2 --data_train DIV2K4x --data_test DIV2K4x --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 14   --patch_size 96   --print_every 100  --argument_scale 1  --light_argument 1  --rgb_range 1   --scale 4  --ressave_path RCAN_Large-PreTrainEX1


### rcan_LARGE   + three dataset with random blur and resize 

python main_denoise.py --model rcan_l --resume -2 --data_train CT0_Random_Cfg1_Down_X4+DeepLesion_Random_Cfg1_Down_X4+LDIC_Random_Cfg1_Down_X4 --data_test CT0_Random_Cfg1_Down_X4+DeepLesion_Random_Cfg1_Down_X4+LDIC_Random_Cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 14   --patch_size 96   --print_every 100  --argument_scale 1  --rgb_range 1   --scale 4  --ressave_path RCAN_Large-EX1

#### pretrain 
python main_denoise.py --model rcan_l --resume 0 --data_train  DIV2K4x --data_test DIV2K4x  --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 14   --patch_size 96   --print_every 100  --argument_scale 1  --rgb_range 1   --scale 4  --pre_train [RCAN_Large-PreTrainEX1]-[DIV2K4x]-[2026-06-04-13-41] --test_only 


### finetune 

python main_denoise.py --model rcan_l --resume 0 --data_train CT0_Random_Cfg1_Down_X4+DeepLesion_Random_Cfg1_Down_X4+LDIC_Random_Cfg1_Down_X4 --data_test CT0_Random_Cfg1_Down_X4+DeepLesion_Random_Cfg1_Down_X4+LDIC_Random_Cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 12   --patch_size 128   --print_every 100  --argument_scale 2  --rgb_range 1   --scale 4  --pre_train [RCAN_Large-PreTrainEX1]-[DIV2K4x]-[2026-06-04-16-42]  --ressave_path RCAN_Large-EX1

python main_denoise.py --model rcan_l --resume 0 --data_train CT0_Random_Cfg1_Down_X4+DeepLesion_Random_Cfg1_Down_X4+LDIC_Random_Cfg1_Down_X4 --data_test CT0_Random_Cfg1_Down_X4+DeepLesion_Random_Cfg1_Down_X4+LDIC_Random_Cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 12   --patch_size 128   --print_every 100  --argument_scale 2  --rgb_range 1   --scale 4  --pre_train [RCAN_Large-EX1]-[CT0_Random_Cfg1_Down_X4]-[2026-06-04-17-43]  --test_only 

### rcan_large  fix ct0 

CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model rcan_l --resume 0 --data_train CT0_BLUR_K5S1_AREA_X4 --data_test CT0_BLUR_K5S1_AREA_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 16   --patch_size 128   --print_every 100  --argument_scale 4  --rgb_range 1   --scale 4  --pre_train [RCAN_Large-EX1]-[CT0_Random_Cfg1_Down_X4]-[2026-06-04-17-43]  --ressave_path RCAN_Large-EX2



CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model rcan_l --resume 0 --data_train CT0_fix_cfg1_Down_X4 --data_test CT0_fix_cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 16   --patch_size 128   --print_every 100  --argument_scale 4  --rgb_range 1   --scale 4  --pre_train [RCAN_Large-EX1]-[CT0_Random_Cfg1_Down_X4]-[2026-06-04-17-43]  --test_only


### rcan_large train with the fix cfg 1 that area downsample and blur kersize5 sigma 1.2    and decrease the patchsize with cuda1

CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model rcan_l --resume 0 --data_train CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --data_test CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 16   --patch_size 128  --print_every 100  --argument_scale 4  --rgb_range 1   --scale 4  --pre_train [RCAN_Large-EX1]-[CT0_Random_Cfg1_Down_X4]-[2026-06-04-17-43]  --ressave_path RCAN_Large-EX3

###  HAT Pre Train 

## memory coat 15GB   BATCH8  PATCH48   ITER10 COST TIME 6S    INFER FPS  3 
CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model hat_simple --resume -2 --data_train DIV2K4x --data_test DIV2K4x --loss 1*L1 --lr 1e-4  --epoch 20 --batch_size  8  --patch_size 48   --print_every 10  --argument_scale 1  --rgb_range 1   --scale 4    --ressave_path HAT-Pretrain


CUDA_VISIBLE_DEVICES=0  python main_denoise.py --model hat_simple --resume 0 --data_train CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --data_test CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 7   --patch_size 64   --print_every 150  --argument_scale 8  --rgb_range 1   --scale 4  --pre_train [HAT-Pretrain]-[DIV2K4x]-[2026-06-06-11-16] --ressave_path HAT-EX3


CUDA_VISIBLE_DEVICES=0  python main_denoise.py --model hat_simple --resume -1 --data_train CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --data_test CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 7   --patch_size 64   --print_every 150  --argument_scale 8  --rgb_range 1   --scale 4  --resload_path [HAT-EX2]-[CT0_fix_cfg1_Down_X4]-[2026-06-06-13-12] 


CUDA_VISIBLE_DEVICES=0  python main_denoise.py --model hat_simple --resume 0 --data_train CT0_BLUR_K5S1_AREA_X4 --data_test CT0_BLUR_K5S1_AREA_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 7   --patch_size 64   --print_every 150  --argument_scale 8  --rgb_range 1   --scale 4  --pre_train [HAT-Pretrain]-[DIV2K4x]-[2026-06-06-11-16] --ressave_path HAT-EX2


### test and infer the vitalhuaman dataset 

CUDA_VISIBLE_DEVICES=0  python main_denoise.py --model hat_simple --resume 0 --data_train CT0_BLUR_K5S1_AREA_X4 --data_test Infer_VitalHuman_AreaDownX4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 7   --patch_size 64   --print_every 150  --argument_scale 8  --rgb_range 1   --scale 4  --pre_train [HAT-EX2]-[CT0_fix_cfg1_Down_X4]-[2026-06-06-13-12] --test_only 



#### rcan     more data train     
CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model rcan --resume 0 --data_train CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --data_test CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 22   --patch_size 128  --print_every 100  --argument_scale 4  --rgb_range 1   --scale 4  --pre_train [RCAN-EX1]-[DIV2K4x]-[2026-05-26-13-08] --ressave_path RCAN-EX3

CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model rcan --resume -1 --data_train CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --data_test CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 22   --patch_size 128  --print_every 100  --argument_scale 4  --rgb_range 1   --scale 4  --resload_path [RCAN-EX3]-[CT0_fix_cfg1_Down_X4]-[2026-06-08-17-30] 
### test and infer the vitalhuaman dataset 

CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model rcan --resume 0 --data_train CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --data_test Infer_VitalHuman_AreaDownX4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 22   --patch_size 128  --print_every 100  --argument_scale 4  --rgb_range 1   --scale 4  --pre_train [RCAN-EX3]-[CT0_fix_cfg1_Down_X4]-[2026-06-08-17-30]   --test_only 


### rcan   test ct0_blurk5s1  

CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model rcan --resume 0 --data_train CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --data_test CT0_BLUR_K5S1_AREA_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 22   --patch_size 128  --print_every 100  --argument_scale 4  --rgb_range 1   --scale 4  --pre_train [RCAN-EX3]-[CT0_fix_cfg1_Down_X4]-[2026-06-08-17-30] --test_only 


## DATA CHANGE : BLUR KERSIZE 1  SIGMA 0      rca_lite train    PRETRIAN   BY   DIV2K   
CUDA_VISIBLE_DEVICES=1 python main_denoise.py --model rcan --resume 0 --data_train LDIC_Fix_cfg2_AreaDownX4+CT0_Fix_cfg2_AreaDownX4+DeepLesion_Fix_cfg2_AreaDownX4 --data_test LDIC_Fix_cfg2_AreaDownX4+CT0_Fix_cfg2_AreaDownX4+DeepLesion_Fix_cfg2_AreaDownX4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 24  --patch_size 128   --print_every 100  --argument_scale 2  --rgb_range 1  --pre_train [RCAN-EX1]-[DIV2K4x]-[2026-05-26-13-08] --scale 4  --ressave_path RCAN-EX4
CUDA_VISIBLE_DEVICES=1 python main_denoise.py --model rcan --resume -1 --data_train LDIC_Fix_cfg2_AreaDownX4+CT0_Fix_cfg2_AreaDownX4+DeepLesion_Fix_cfg2_AreaDownX4 --data_test LDIC_Fix_cfg2_AreaDownX4+CT0_Fix_cfg2_AreaDownX4+DeepLesion_Fix_cfg2_AreaDownX4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 24  --patch_size 128   --print_every 100  --argument_scale 2  --rgb_range 1  --resload_path [RCAN-EX4]-[LDIC_Fix_cfg2_AreaDownX4]-[2026-06-12-11-17] --scale 4  


### DATA CHANGE : BLUR KERSIZE 1  SIGMA 0     hat    train    PRETRIAN   BY   DIV2K office        


CUDA_VISIBLE_DEVICES=0 python main_denoise.py --model hat_simple --resume 0 --data_train LDIC_Fix_cfg2_AreaDownX4+CT0_Fix_cfg2_AreaDownX4+DeepLesion_Fix_cfg2_AreaDownX4 --data_test LDIC_Fix_cfg2_AreaDownX4+CT0_Fix_cfg2_AreaDownX4+DeepLesion_Fix_cfg2_AreaDownX4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 8  --patch_size 64   --print_every 100  --argument_scale 2  --rgb_range 1  --pre_train [HAT-Office]-[DIV2K+FIL2K]-[2024-10-10] --scale 4  


CUDA_VISIBLE_DEVICES=0 python main_denoise.py --model hat_simple --resume -1 --data_train LDIC_Fix_cfg2_AreaDownX4+CT0_Fix_cfg2_AreaDownX4+DeepLesion_Fix_cfg2_AreaDownX4 --data_test LDIC_Fix_cfg2_AreaDownX4+CT0_Fix_cfg2_AreaDownX4+DeepLesion_Fix_cfg2_AreaDownX4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 8  --patch_size 64   --print_every 100  --argument_scale 2  --rgb_range 1  --resload_path [HAT-EX4]-[LDIC_Fix_cfg2_AreaDownX4]-[2026-06-12-15-02] --scale 4  
