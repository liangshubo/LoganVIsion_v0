CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model rcan_l --resume 0 --data_train CT0_BLUR_K5S1_AREA_X4 --data_test CT0_BLUR_K5S1_AREA_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 16   --patch_size 128   --print_every 100  --argument_scale 8  --rgb_range 1   --scale 4  --pre_train [RCAN_Large-EX1]-[CT0_Random_Cfg1_Down_X4]-[2026-06-04-17-43]  --ressave_path RCAN_Large-EX2

### rcan_large train with the fix cfg 1 that area downsample and blur kersize5 sigma 1.2    and decrease the patchsize with cuda1

CUDA_VISIBLE_DEVICES=1  python main_denoise.py --model rcan_l --resume 0 --data_train CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4  --data_test CT0_fix_cfg1_Down_X4+DeepLesion_fix_cfg1_Down_X4 --loss 1*L1 --lr 1e-4  --epoch 100 --batch_size 16   --patch_size 128  --print_every 100  --argument_scale 4  --rgb_range 1   --scale 4  --pre_train [RCAN_Large-EX1]-[CT0_Random_Cfg1_Down_X4]-[2026-06-04-17-43]  --ressave_path RCAN_Large-EX3
