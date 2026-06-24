import torch
import argparse


def build_model(scale=4):
    """
    在这里构建你的模型结构。
    重点：pth 如果是 state_dict，必须先有模型结构，再 load_state_dict。
    """

    # ===== 示例 1：如果你的模型可以这样导入 =====
    # from model.rcan import RCAN
    # from option import args
    # args.scale = scale
    # model = RCAN(args)

    # ===== 示例 2：如果你有 make_model(args) =====
    from option import args
    from model.hat_simple import make_model

    args.scale = scale
    model = make_model(args)

    return model


def load_pth_to_model(model, pth_path, device='cuda:0',strict=False):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(pth_path, map_location=device)

    # ==============================
    # 1. 兼容不同 checkpoint 格式
    # ==============================
    if isinstance(checkpoint, dict):
        if "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
            print("使用 checkpoint['params_ema']")
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
            print("使用 checkpoint['params']")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("使用 checkpoint['state_dict']")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("使用 checkpoint['model']")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("使用 checkpoint['model_state_dict']")
        else:
            state_dict = checkpoint
            print("使用 checkpoint 本身作为 state_dict")
    else:
        raise TypeError("checkpoint 格式不对，应该是 dict 或 state_dict")

    # ==============================
    # 2. 去掉 module. 前缀
    # ==============================
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    state_dict = new_state_dict

    # 灰度加权系数
    rgb_weight = torch.tensor(
        [0.299, 0.587, 0.114],
        dtype=torch.float32,
        device=device
    )

    # ==============================
    # 3. 处理 conv_first.weight
    # [180, 3, 3, 3] -> [180, 1, 3, 3]
    # ==============================
    if "conv_first.weight" in state_dict:
        w = state_dict["conv_first.weight"].to(device)

        if w.ndim == 4 and w.shape[1] == 3 and model.conv_first.weight.shape[1] == 1:
            print(f"转换 conv_first.weight: {tuple(w.shape)} -> ", end="")

            # rgb_weight: [3] -> [1, 3, 1, 1]
            weight = rgb_weight.view(1, 3, 1, 1)

            # 加权求和输入通道维度
            w_gray = (w * weight).sum(dim=1, keepdim=True)

            state_dict["conv_first.weight"] = w_gray.cpu() if model.conv_first.weight.device.type == "cpu" else w_gray

            print(tuple(w_gray.shape))

    # ==============================
    # 4. 处理 conv_last.weight
    # [3, 64, 3, 3] -> [1, 64, 3, 3]
    # ==============================
    if "conv_last.weight" in state_dict:
        w = state_dict["conv_last.weight"].to(device)

        if w.ndim == 4 and w.shape[0] == 3 and model.conv_last.weight.shape[0] == 1:
            print(f"转换 conv_last.weight: {tuple(w.shape)} -> ", end="")

            # rgb_weight: [3] -> [3, 1, 1, 1]
            weight = rgb_weight.view(3, 1, 1, 1)

            # 加权求和输出通道维度
            w_gray = (w * weight).sum(dim=0, keepdim=True)

            state_dict["conv_last.weight"] = w_gray.cpu() if model.conv_last.weight.device.type == "cpu" else w_gray

            print(tuple(w_gray.shape))

    # ==============================
    # 5. 处理 conv_last.bias
    # [3] -> [1]
    # ==============================
    if "conv_last.bias" in state_dict:
        b = state_dict["conv_last.bias"].to(device)

        if b.ndim == 1 and b.shape[0] == 3 and model.conv_last.bias.shape[0] == 1:
            print(f"转换 conv_last.bias: {tuple(b.shape)} -> ", end="")

            b_gray = (b * rgb_weight).sum(dim=0, keepdim=True)

            state_dict["conv_last.bias"] = b_gray.cpu() if model.conv_last.bias.device.type == "cpu" else b_gray

            print(tuple(b_gray.shape))

    # ==============================
    # 6. 加载权重
    # ==============================
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    print("预训练权重加载完成")
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    return model


def convert_pth_to_pt(
    pth_path,
    pt_path,
    scale=4,
    input_h=120,
    input_w=160,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = build_model(scale=scale)
    model = model.to(device)

    model = load_pth_to_model(model, pth_path, device)
    model.eval()

    dummy_input = torch.randn(1, 1, input_h, input_w).to(device)
    print(model)
    with torch.no_grad():
        # trace 方式，适合 RCAN / SwinIR / HAT 这类常规前向推理模型
        #traced_model = torch.jit.trace(model, dummy_input)

        # 再跑一次检查
        out = model(dummy_input)
        print("Input shape :", dummy_input.shape)
        print("Output shape:", out.shape)

        torch.save(model.state_dict(),pt_path)

    print(f"转换完成：{pth_path} -> {pt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pth", type=str,  default="/home/liangshubo/Project/CTSR/experiment/[HAT-Office]-[DIV2K+FIL2K]-[2024-10-10]/PTH/HAT_SRx4.pth", help="输入 pth 权重路径")
    parser.add_argument("--pt", type=str,  default="/home/liangshubo/Project/CTSR/experiment/[HAT-Office]-[DIV2K+FIL2K]-[2024-10-10]/model/HAT_SRx4.pt", help="输出 pt 模型路径")
    parser.add_argument("--scale", type=int, default=4, help="超分倍率")
    parser.add_argument("--input_h", type=int, default=120, help="测试输入高度")
    parser.add_argument("--input_w", type=int, default=160, help="测试输入宽度")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")

    args = parser.parse_args()

    convert_pth_to_pt(
        pth_path=args.pth,
        pt_path=args.pt,
        scale=args.scale,
        input_h=args.input_h,
        input_w=args.input_w,
        device=args.device
    )