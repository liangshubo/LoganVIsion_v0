#python main_denoise.py --model baselineunet --resume -2 --data_train thyroid_samsung_level2 --data_test thyroid_samsung_level2 --loss 1*L1+1*Percept --lr 1e-4  --epoch 500 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Research-sam2
#python main_denoise.py --model baselineunet --resume -2 --data_train thyroid_samsung_level3 --data_test thyroid_samsung_level3 --loss 1*L1+1*Percept --lr 1e-4  --epoch 500 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Research-sam3
#python main_denoise.py --model baselineunet --resume -2 --data_train thyroid_samsung_level4 --data_test thyroid_samsung_level4 --loss 1*L1+1*Percept --lr 1e-4  --epoch 500 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Research-sam4
#python main_denoise.py --model baselineunet --resume -2 --data_train thyroid_samsung_level5 --data_test thyroid_samsung_level5 --loss 1*L1+1*Percept --lr 1e-4  --epoch 500 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Research-sam5
#
#
#
#python main_denoise.py --model baselineunet --resume -1 --data_train thyroid_samsung_level3 --data_test thyroid_samsung_level3 --loss 1*L1+1*Percept --lr 1e-4  --epoch 1500 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Research-sam3]-[thyroid_samsung_level3]

#
#python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_v1_1 --data_test n20_thyroid_v1_1 --loss 1*L1+1*Percept --lr 1e-4  --epoch 1000 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v1_1
#python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_v1_2 --data_test n20_thyroid_v1_2 --loss 1*L1+1*Percept --lr 1e-4  --epoch 1000 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v1_1

#python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam3_cv2 --data_test n20_thyroid_sam3_cv2 --loss 1*L1+1*Percept --lr 1e-4  --epoch 1000 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_1
#python main_denoise.py --model baselineunet --resume -1 --data_train n20_thyroid_sam3_cv2_sr --data_test n20_thyroid_sam3_cv2_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 1000 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Deploy-v2_1]-[n20_thyroid_sam3_cv2_sr]


# style2 sam thyroid ->cv -> sr
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam2_cv1_sr --data_test n20_thyroid_sam2_cv1_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_2
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam2_cv2_sr --data_test n20_thyroid_sam2_cv2_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_2
# python main_denoise.py --model baselineunet --resume -1 --data_train n20_thyroid_sam3_cv3_sr --data_test n20_thyroid_sam3_cv3_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Deploy-v2_2]-[n20_thyroid_sam3_cv3_sr]-[2024-07-29-08-43]
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam4_cv4_sr --data_test n20_thyroid_sam4_cv4_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_2

# # style4 sam thyroid ->cv ->srblur   12 在这里4090 的机器训练   345 在 2080 上进行训练


# python main_denoise.py --model baselineunet --resume -1 --data_train n20_thyroid_sam2_cv1_srblur --data_test n20_thyroid_sam2_cv1_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Deploy-v2_4]-[n20_thyroid_sam2_cv1_srblur]-[2024-08-01-00-50]
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam2_cv2_srblur --data_test n20_thyroid_sam2_cv2_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_4


# # 0802  sam4 的训练 作为第一个档位的效果
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam4 --data_test n20_thyroid_sam4 --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 300   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v3




# # light baseline
# python main_denoise.py --model baselineunet_light --resume -2 --data_train n20_thyroid_sam2_cv1_sr --data_test n20_thyroid_sam2_cv1_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline_light-Thyroid_plus-Deploy-v2_2
# python main_denoise.py --model baselineunet_light --resume -2 --data_train n20_thyroid_sam2_cv2_sr --data_test n20_thyroid_sam2_cv2_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline_light-Thyroid_plus-Deploy-v2_2
# python main_denoise.py --model baselineunet_light --resume -2 --data_train n20_thyroid_sam3_cv3_sr --data_test n20_thyroid_sam3_cv3_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline_light-Thyroid_plus-Deploy-v2_2
# python main_denoise.py --model baselineunet_light --resume -2 --data_train n20_thyroid_sam4_cv4_sr --data_test n20_thyroid_sam4_cv4_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline_light-Thyroid_plus-Deploy-v2_2


# # python main_denoise.py --model baselineunet --resume -1 --data_train n20_thyroid_sam3_cv2_sr --data_test n20_thyroid_sam3_cv2_sr --loss 1*L1+1*Percept --lr 1e-4  --epoch 1000 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Deploy-v2_1]-[n20_thyroid_sam3_cv2_sr]

# #


# # style4 sam thyroid ->cv -> srblur 
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam3_cv2_srblur --data_test n20_thyroid_sam3_cv2_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_4
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam3_cv3_srblur --data_test n20_thyroid_sam3_cv3_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_4
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam4_cv4_srblur --data_test n20_thyroid_sam4_cv4_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_4
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam2_cv1_srblur --data_test n20_thyroid_sam2_cv1_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_4
# python main_denoise.py --model baselineunet --resume -2 --data_train n20_thyroid_sam2_cv2_srblur --data_test n20_thyroid_sam2_cv2_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 340   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --ressave_path Baseline-Thyroid_plus-Deploy-v2_4





# ##   -------------20240802 ----style4 resume and sam4 train --------------------------
python main_denoise.py --model baselineunet --resume -1 --data_train n20_thyroid_sam3_cv3_srblur --data_test n20_thyroid_sam3_cv3_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 32  --patch_size 300   --print_every 300  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Deploy-v2_4]-[n20_thyroid_sam3_cv3_srblur]-[2024-07-30-05-06]
python main_denoise.py --model baselineunet --resume -1 --data_train n20_thyroid_sam3_cv2_srblur --data_test n20_thyroid_sam3_cv2_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 32  --patch_size 300   --print_every 300  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Deploy-v2_4]-[n20_thyroid_sam3_cv2_srblur]-[2024-07-29-08-45]


# python main_denoise.py --model baselineunet --resume -1 --data_train n20_thyroid_sam4_cv4_srblur --data_test n20_thyroid_sam4_cv4_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 32  --patch_size 280   --print_every 300  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Deploy-v2_4]-[n20_thyroid_sam4_cv4_srblur]-[2024-07-30-06-49]



# python main_denoise.py --model baselineunet --resume -1 --data_train n20_thyroid_sam4 --data_test n20_thyroid_sam4 --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 48  --patch_size 300   --print_every 100  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Deploy-v3]-[n20_thyroid_sam4]-[2024-08-02-14-04]
# python main_denoise.py --model baselineunet --resume -1 --data_train n20_thyroid_sam2_cv1_srblur --data_test n20_thyroid_sam2_cv1_srblur --loss 1*L1+1*Percept --lr 1e-4  --epoch 600 --batch_size 32  --patch_size 280   --print_every 300  --argument_scale 8  --light_argument 1  --rgb_range 1  --resload_path [Baseline-Thyroid_plus-Deploy-v2_4]-[n20_thyroid_sam2_cv1_srblur]-[2024-08-01-00-50]
