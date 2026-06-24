import torch
import cv2
import torchvision
import argparse
import model
from importlib import import_module
import os
import time
# import SimpleITK as sitk

import glob

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

project_path  = os.path.dirname(os.path.dirname(__file__))
#print(project_path)
def get_args():
    parser = argparse.ArgumentParser(description='Thermal and Rail SR')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    parser.add_argument('--model', default='Unet',
                        help='model name')
    parser.add_argument('--pre_train', type=str, default=None,
                        help='pre-trained model directory')
    parser.add_argument('--iterates', type=int, default=30,
                        help='iterates number ')

    parser.add_argument('--input_data', type=str, default=None,
                        help='single image  directory')

    parser.add_argument('--input_path', type=str, default=None,
                        help='more image directory')

    parser.add_argument('--project_path', type=str, default=project_path,
                        help='project_path ')

    parser.add_argument('--save_path', type=str, default=None,
                        help='save_path ')
    parser.add_argument('--sw_mode', type=str, default="S",
                        help='save_path ')

    parser.add_argument('--resume', type=int, default=0,
                        help='control the load model is best or lastest or other ')

    parser.add_argument('--input_size', type=int, default=256,
                        help='control the input data crop patch (592,720), if patch_size != None it will be crop shuffle 作为正方形')

    args = parser.parse_args()
    return args


def tensor2array(tensor, rgb_range):
    tensor = tensor.squeeze(0)
    array = tensor.squeeze(0).cpu().numpy() * (255 / rgb_range)
    return array


def control_pretrain(resume: int) -> str:
    if resume == 0:
        return "best"
    elif resume == -1:
        return "latest"
    else:
        return resume


def inference_single_image(args):
    if args.cpu:
        # ----加载模型 #导入模型 ，所以这里要注意的是都是小写的
        module = import_module('model.' + args.model.lower())  # 导入模型 ，所以这里要注意的是都是小写的
        model = module.make_model(args)
        model = model.cpu()
        # ----预训练参数路径
        if args.pre_train:
            pre_train_model_path = os.path.join(args.project_path, 'experiment', args.pre_train, "model")
            model.load_state_dict(
                torch.load(pre_train_model_path + f"/model_{control_pretrain(args.resume)}.pt", map_location='cpu'),
                strict=False)

        print("[----> Model inference in ", next(model.parameters()).device, "<----]")

        # ------加载输入数据 ----
        input_data = args.input_data
        # -------输入数据路径与文件名分离
        path, nameext = os.path.split(input_data)
        # -------数据读取与预处理
        input = torch.tensor(cv2.imread(input_data, 0)).unsqueeze(0).unsqueeze(0) / 255
        INPUT = input.cpu()

        print("[----> Data Input device {} <----]".format(INPUT.device))
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # -------模型前向处理---------
        avg_time = 0
        with torch.no_grad():
            # starter.record()
            start = time.time()
            output = model(INPUT)
            # ------结果保存-----
            output = tensor2array(output, 1)
            cv2.imwrite(os.path.join(args.save_path, nameext), output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # ender.record()
            end = time.time()
            # cur_time = starter.elapsed_time(ender)
            cur_time = (end - start) * 1000
            avg_time += cur_time
            print("Iter [{}/{}] : {:.5f}ms".format(iter, args.iterates, cur_time))

    else:
        # ----加载模型 #导入模型 ，所以这里要注意的是都是小写的
        module = import_module('model.' + args.model.lower())
        model = module.make_model(args)
        # ----预训练参数路径
        if args.pre_train:
            pre_train_model_path = os.path.join(args.project_path, 'experiment', args.pre_train, "model")
            model.load_state_dict(torch.load(pre_train_model_path + f"/model_{control_pretrain(args.resume)}.pt"),
                                  strict=False)
        model = model.cuda()
        print("[----> Model inference in ", next(model.parameters()).device, "<----]")

        # ------加载输入数据 ----
        input_data = args.input_data
        # -------输入数据路径与文件名分离
        path, nameext = os.path.split(input_data)
        # -------数据读取与预处理
        input = torch.tensor(cv2.imread(input_data, 0)).unsqueeze(0).unsqueeze(0) / 255
        INPUT = input.cuda()

        print("[----> Data Input device {} <----]".format(INPUT.device))
        # -------预热
        print("[----> Hot GPU ... <----]")
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(5):
            _ = model(INPUT.cuda())
        # -------模型前向处理---------
        avg_time = 0
        with torch.no_grad():
            starter.record()
            output = model(INPUT)
            # ------结果保存-----
            output = tensor2array(output, 1)
            cv2.imwrite(os.path.join(args.save_path, nameext), output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            ender.record()
            torch.cuda.synchronize()
            cur_time = starter.elapsed_time(ender)
            # cur_time = (end-start)*1000
            avg_time += cur_time
            print("Time [{}/{}] : {:.5f}ms".format(0, args.iterates, cur_time))

    return 0


def inference_multi_image(args):
    if args.cpu:
        # ----加载模型 #导入模型 ，所以这里要注意的是都是小写的
        module = import_module('model.' + args.model.lower())  # 导入模型 ，所以这里要注意的是都是小写的
        model = module.make_model(args).cpu()
        # ----预训练参数路径
        if args.pre_train:
            pre_train_model_path = os.path.join(args.project_path, 'experiment', args.pre_train, "model")
            model.load_state_dict(
                torch.load(pre_train_model_path + f"/model_{control_pretrain(args.resume)}.pt", map_location='cpu'),
                strict=False)
        model = model.cpu()
        print("[----> Model inference in ", next(model.parameters()).device, "<----]")

        name_list = os.listdir(args.input_path)
        count = 1
        for nameext in name_list:
            # -------输入数据路径与文件名分离
            input_data = os.path.join(args.input_path, nameext)

            # -------输入数据路径与文件名分离

            # -------数据读取与预处理
            input = torch.tensor(cv2.imread(input_data, 0)).unsqueeze(0).unsqueeze(0) / 255
            INPUT = input.cpu()

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # -------模型前向处理---------
            avg_time = 0
            with torch.no_grad():
                # starter.record()
                start = time.time()
                #    model input idx

                output = model(INPUT)

                # ------结果保存-----
                output = tensor2array(output, 1)
                cv2.imwrite(os.path.join(args.save_path, nameext), output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # ender.record()
                end = time.time()
                # cur_time = starter.elapsed_time(ender)
                cur_time = (end - start) * 1000
                avg_time += cur_time
                print("Image[{}/{}] : {:.5f}ms".format(count, len(name_list), cur_time))
            count += 1
    else:
        # ----加载模型 #导入模型 ，所以这里要注意的是都是小写的
        module = import_module('model.' + args.model.lower())
        model = module.make_model(args)
        # ----预训练参数路径
        if args.pre_train:
            pre_train_model_path = os.path.join(args.project_path, 'experiment', args.pre_train, "model")
            model.load_state_dict(torch.load(pre_train_model_path + f"/model_{control_pretrain(args.resume)}.pt"),
                                  strict=False)
        model = model.cuda()
        print("[----> Model inference in ", next(model.parameters()).device, "<----]")

        # ------加载文件夹内输入数据 ----

        name_list = os.listdir(args.input_path)
        count = 1
        for nameext in name_list:
            # -------输入数据路径与文件名分离
            input_data = os.path.join(args.input_path, nameext)
            # -------数据读取与预处理
            input = torch.tensor(cv2.imread(input_data, 0)).unsqueeze(0).unsqueeze(0) / 255

            INPUT = input.cuda()
            # print("[----> Data Input device {} <----]".format(INPUT.device))
            # -------预热
            # print("[----> Hot GPU ... <----]")
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            #for _ in range(5):
            #    _ = model(INPUT.cuda())
            # -------模型前向处理---------
            avg_time = 0
            with torch.no_grad():
                starter.record()
                # 　－－－－－－－－
                output = model(INPUT)
                #  - - - - - - - -
                # ------结果保存-----

                # mask_tensor = biliary_mask(output)
                # mask_array = tensor2array( mask_tensor,1)
                # mask_array  = mcc_edge(mask_array*255)/255

                output = tensor2array(output, 1)
                cv2.imwrite(os.path.join(args.save_path, nameext), output, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                # cv2.imwrite(os.path.join(args.save_path,nameext), mask_array,[cv2.IMWRITE_PNG_COMPRESSION,0])
                ender.record()
                torch.cuda.synchronize()
                cur_time = starter.elapsed_time(ender)
                # cur_time = (end-start)*1000
                avg_time += cur_time
                print("Image[{}/{}] : {:.5f}ms".format(count, len(name_list), cur_time))
            count += 1
    return model





if __name__ == '__main__':

    import datetime
    file_path  = os.path.dirname(__file__)
    log = open(file_path+"/demo_log.txt","a")
    import glob

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # 1. 指定模型
    args.model = "baselineunet_mid"#"baselineunet"#"baselineunet"#"baselineunet_mid"
    args.resume = 0
    # args.cpu = True
    # [Baseline-ThyEX02]-[N20Thyroid_A_MapDown_CV1]-[2024-02-19-14-24]
    # [Baseline-ThyEX02]-[N20Thyroid_A_MapDown_CV2]-[2024-02-19-22-40]
    # [Baseline-ThyEX02]-[N20Thyroid_A_MapDown_CV3]-[2024-02-22-23-43]
    # [Baseline-ThyEX02]-[N20Thyroid_A_MapDown_CV4]-[2024-02-23-09-11]

    args.pre_train = "[BaselineMid-MSK_plus-Research-SAMLAMSK-lv1]-[samsung_msk_la22_lv1]-[2024-09-03-14-58]"

    single_image = False
    if single_image == True:

        args.input_data = "/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/src/demoimage/0204FR06/0204FR060.png"
        args.save_path = "/ultrasound/LiangShubo/DenoiseCode/LumenEnhance/src/demoimage/mask_result"
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        model = inference_single_image(args)
    else:
        # 3.，指定文件夹路径
        args.input_path = r"/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/dataset_process_926/plan2/GAN_SCI_02"

        # 4. 指定保存路径
        
        args.save_path = "/home/ubuntu4090/4T_disk/liangshubo/MSK_Plus/dataset/rawdata/n20_sl14_3h_msk_process/dataset_process_926/plan2/GAN_SCI_02->SAM_MSK_LV1"
        # 5. 运行
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        model = inference_multi_image(args)
        log.write(f"[-------------------{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}--------------------]")
        log.write(f"\n pretrain:[{args.pre_train}] \n"+f"src_path :[{args.input_path}] \n"+f"dst_path :[{args.save_path}]")
    #

    #
    # model = model.cuda()
    # from thop import clever_format
    # from thop import profile
    # input = torch.randn([1, 1, 576,768]).cuda()
    # flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print('flops : {}'.format(flops))
    # print('params : {}'.format(params))
    # print('Total params: %.4fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

# python inference.py --cpu --model CBDNet --pre_train [CBDNet-P256L1B16RGB1]-[denoisecv5]-[2023-09-01-00-59]
# Python CPU Inference time: 729.183502ms ,FPS: 1.3713969076188322
# python inference.py --model CBDNet --pre_train [CBDNet-P256L1B16RGB1]-[denoisecv5]-[2023-09-01-00-59]
# Python GPU Inference time: 29.455276ms ,FPS: 33.949775487860364
# python inference.py --model RIDNet --pre_train [RIDNet-P256L1B16]-[denoisecv3]-[2023-08-29-02-44]
# Python GPU Inference time: 94.437861ms ,FPS: 10.588973406331442
# python inference.py --model RIDNet --pre_train [RIDNet-P256L1B16]-[denoisecv3]-[2023-08-29-02-44] --cpu
# Python CPU Inference time: 2191.887199ms ,FPS: 0.45622785721132897
# python inference.py --model baseline --cpu
# Python CPU Inference time: 319.793439ms ,FPS: 3.127019
# python inference.py --model baseline
# Python GPU Inference time: 9.671605ms ,FPS: 103.395451


# torch.randn(1,1,576,768)
# python inference.py --model baselineunet
# Python GPU Inference time: 17.120564ms ,FPS: 58.409289 flops : 22.723G  params : 816.065K
# python inference.py --model baselineunet --cpu
# Python CPU Inference time: 397.805556ms ,FPS: 2.513791 flops : 22.723G  params : 816.065K

# python inference.py --model baselinecbdnet
# Python GPU Inference time: 14.738573ms ,FPS: 67.849175 flops : 71.615G  params : 4.368M
# python inference.py --model baselinecbdnet --cpu
# Python CPU Inference time: 358.365719ms ,FPS: 2.790445 flops : 71.615G  params : 4.368M

# python inference.py --model cbdnet
# Python GPU Inference time: 42.835006ms ,FPS: 23.345392 flops : 260.802G params : 4.362M
# python inference.py --model cbdnet --cpu
# Python CPU Inference time: 1155.511761ms ,FPS: 0.865417 flops : 260.802G params : 4.362M































