import torch
import cv2
import torchvision
import argparse
import model
from importlib import import_module
import os
import time
import SimpleITK as sitk

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_args():
    parser = argparse.ArgumentParser(description='Thermal and Rail SR')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    parser.add_argument('--model', default='Unet',
                        help='model name')
    parser.add_argument('--pre_train', type=str, default=None,
                        help='pre-trained model directory')
    parser.add_argument('--iterates', type=int, default=500,
                        help='iterates number ')

    args = parser.parse_args()
    return args


'''
def get_data(args):
    dataset_path = 
'''


def inference(args):
    RANDOM_INPUT = torch.randn(1, 1, 600, 800)
    # RANDOM_INPUT = torch.randn(1,1,454,672)
    if args.cpu:
        # 加载模型，
        module = import_module('model.' + args.model.lower())  # 导入模型 ，所以这里要注意的是都是小写的
        model = module.make_model(args)
        if args.pre_train:
            pre_train_model_path = os.path.join('../srcv2.0', 'experiment', args.pre_train, "model")
            model.load_state_dict(torch.load(pre_train_model_path + "/model_best.pt"),
                                  strict=False)
        model = model.cpu()
        print("[----> Model inference in ", next(model.parameters()).device, "<----]")
        RANDOM_INPUT = RANDOM_INPUT.cpu()
        print("[----> Data Input device {} <----]".format(RANDOM_INPUT.device))
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # 计算推理时间
        avg_time = 0
        with torch.no_grad():
            for iter in range(args.iterates):
                # starter.record()
                start = time.time()
                _ = model(RANDOM_INPUT)
                # ender.record()
                _ = _.cpu()
                end = time.time()
                # cur_time = starter.elapsed_time(ender)
                cur_time = (end - start) * 1000
                avg_time += cur_time
                print("Iter [{}/{}] : {:.5f}ms".format(iter, args.iterates, cur_time))
            avg_time = avg_time / args.iterates
            print("Python CPU Inference time: {:.6f}ms ,FPS: {:.6f}".format(avg_time, 1000 / avg_time))
    else:
        # 加载模型
        module = import_module('model.' + args.model.lower())  # 导入模型 ，所以这里要注意的是都是小写的
        model = module.make_model(args)
        if args.pre_train:
            pre_train_model_path = os.path.join('../srcv2.0', 'experiment', args.pre_train, "model")
            model.load_state_dict(torch.load(pre_train_model_path + "/model_best.pt"),
                                  strict=False)
        model = model.cuda()
        print("[----> Model inference in ", next(model.parameters()).device, "<----]")
        RANDOM_INPUT = RANDOM_INPUT
        print("[----> Data Input device {} <----]".format(RANDOM_INPUT.device))
        # 预热
        print("[----> Hot GPU ... <----]")
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(50):
            _ = model(RANDOM_INPUT.cuda())

        avg_time = 0
        with torch.no_grad():
            for iter in range(args.iterates):
                # starter.record()
                start = time.time()
                RANDOM_INPUT = RANDOM_INPUT.cuda()
                _ = model(RANDOM_INPUT)
                # ender.record()
                # _ = _.cpu()
                torch.cuda.synchronize()
                end = time.time()
                # cur_time = starter.elapsed_time(ender)
                cur_time = (end - start) * 1000
                avg_time += cur_time
                print("Iter [{}/{}] : {:.5f}ms".format(iter, args.iterates, cur_time))
            avg_time = avg_time / args.iterates
            print("Python GPU Inference time: {:.6f}ms ,FPS: {:.6f}".format(avg_time, 1000 / avg_time))

    return model


if __name__ == '__main__':
    args = get_args()
    args.model = "rcan"
    args.cpu = False
    model = inference(args)
    args.pre_train = None  # "[Baseline_Light_unet-CEX1-Finetune]-[Pretrainfinetunecv3]-[2023-11-02-01-11]"
    model = model.cuda()
    from thop import clever_format
    from thop import profile

    input = torch.randn([1, 1, 600, 800]).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops : {}'.format(flops))
    print('params : {}'.format(params))
    print('Total params: %.4fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

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

# Python  inference.py --model baselineunet --cpu  # used unetlight
# Python CPU Inference time: 75.170676ms ,FPS: 13.303060   flops : 4.279G   params : 726.073K   Total params: 0.7261M