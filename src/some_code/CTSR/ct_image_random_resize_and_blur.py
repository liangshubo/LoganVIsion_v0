import os
import cv2
import numpy as np

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Sequence


# =========================================================
# 类型别名
# =========================================================
InterpolationProb = Tuple[int, float]


# =========================================================
# 1. 随机退化配置
# =========================================================
@dataclass(frozen=True)
class RandomDegradationConfig:
    """
    随机退化配置。

    参数
    ----
    enabled:
        是否开启随机退化。
    blur_prob:
        使用 Gaussian blur 的概率。
        例如 1.0 表示每张图都模糊；
        0.7 表示 70% 的图像会模糊，30% 不模糊。
    blur_kernel_choices:
        可选的模糊核大小。
        必须是正奇数，例如 (3, 5, 7)。
    blur_sigma_range:
        sigma 随机范围，例如 (0.2, 1.2)。
    interpolation_choices:
        下采样插值方法及其概率。
        例如：
        (
            (cv2.INTER_AREA, 0.50),
            (cv2.INTER_LINEAR, 0.25),
            (cv2.INTER_CUBIC, 0.25),
        )

    seed:
        随机种子。设置后可复现。
    """
    enabled: bool = False
    blur_prob: float = 0.7
    blur_kernel_choices: Tuple[int, ...] = (3, 5, 7)
    blur_sigma_range: Tuple[float, float] = (0.2, 1.2)
    interpolation_choices: Tuple[InterpolationProb, ...] = (
        (cv2.INTER_AREA, 0.50),
        (cv2.INTER_LINEAR, 0.25),
        (cv2.INTER_CUBIC, 0.25),
    )
    seed: Optional[int] = 2026


# =========================================================
# 2. 实际采样到的退化参数
# =========================================================
@dataclass(frozen=True)
class DegradationParams:
    blur_kernel: int
    blur_sigma: float
    interpolation: int
    random_enabled: bool

# =========================================================
# 3. 工具函数：检查模糊核
# =========================================================
def ensure_odd_kernel(kernel: int) -> int:
    """
    保证 GaussianBlur 的 kernel 是正奇数。
    """
    kernel = int(kernel)
    if kernel <= 1:
        return 1
    if kernel % 2 == 0:
        kernel += 1
    return kernel


# =========================================================
# 4. 工具函数：插值方法名字，仅用于打印日志
# =========================================================
def interpolation_name(interpolation: int) -> str:
    name_map = {
        cv2.INTER_NEAREST: "INTER_NEAREST",
        cv2.INTER_LINEAR: "INTER_LINEAR",
        cv2.INTER_CUBIC: "INTER_CUBIC",
        cv2.INTER_AREA: "INTER_AREA",
        cv2.INTER_LANCZOS4: "INTER_LANCZOS4",
    }
    return name_map.get(interpolation, f"UNKNOWN_{interpolation}")


# =========================================================
# 5. 工具函数：带概率采样插值方法
# =========================================================
def sample_interpolation(
        choices: Sequence[InterpolationProb],
        rng: np.random.Generator
) -> int:
    """
    根据给定概率随机选择插值方法。
    choices = [ INTER_AREA , 0.5 ]
    """
    if len(choices) == 0:
        raise ValueError("interpolation_choices 不能为空。")
    # 方法
    methods: List[int] = [int(item[0]) for item in choices]
    # 概率
    probs = np.array([float(item[1]) for item in choices], dtype=np.float64)

    if np.any(probs < 0):
        raise ValueError("interpolation_choices 中的概率不能为负数。")

    prob_sum = float(probs.sum())

    if prob_sum <= 0:
        raise ValueError("interpolation_choices 的概率总和必须大于 0。")

    probs = probs / prob_sum

    index = int(rng.choice(len(methods), p=probs))
    return methods[index]


# =========================================================
# 6. 工具函数：采样随机退化参数
# =========================================================
def sample_degradation_params(
        fixed_blur_kernel: int,
        fixed_blur_sigma: float,
        fixed_interpolation: int,
        random_cfg: Optional[RandomDegradationConfig],
        rng: np.random.Generator
) -> DegradationParams:
    """
    根据是否开启随机退化，返回本张图像的退化参数。
    """
    if random_cfg is None or not random_cfg.enabled:
        return DegradationParams(
            blur_kernel=ensure_odd_kernel(fixed_blur_kernel),
            blur_sigma=float(fixed_blur_sigma),
            interpolation=int(fixed_interpolation),
            random_enabled=False
        )

    if not (0.0 <= random_cfg.blur_prob <= 1.0):
        raise ValueError("blur_prob 必须在 [0, 1] 范围内。")

    # 是否使用 blur
    use_blur = bool(rng.random() < random_cfg.blur_prob)

    if use_blur:
        if len(random_cfg.blur_kernel_choices) == 0:
            raise ValueError("blur_kernel_choices 不能为空。")

        kernel = int(rng.choice(random_cfg.blur_kernel_choices))
        kernel = ensure_odd_kernel(kernel)

        sigma_min, sigma_max = random_cfg.blur_sigma_range
        sigma_min = float(sigma_min)
        sigma_max = float(sigma_max)

        if sigma_min < 0 or sigma_max < 0:
            raise ValueError("blur_sigma_range 不能为负数。")

        if sigma_min > sigma_max:
            raise ValueError("blur_sigma_range 必须满足 min <= max。")

        sigma = float(rng.uniform(sigma_min, sigma_max))
    else:
        kernel = 1
        sigma = 0.0

    interpolation = sample_interpolation(
        choices=random_cfg.interpolation_choices,
        rng=rng
    )

    return DegradationParams(
        blur_kernel=kernel,
        blur_sigma=sigma,
        interpolation=interpolation,
        random_enabled=True
    )


# =========================================================
# 7. padding：保证能被 scale 整除
# =========================================================
def pad_to_divisible(
        image: np.ndarray,
        scale: float
) -> Tuple[np.ndarray, bool]:
    """
    对图像右侧和下侧进行 padding，使其尺寸能被 downsample 倍率整除。

    例如：
    scale = 0.25 表示下采样到 1/4。
    divisor = 4。
    图像 H/W 需要能被 4 整除。
    """
    if scale <= 0 or scale >= 1:
        raise ValueError("scale 应该在 (0, 1) 范围内，例如 0.25 表示 4× SR。")

    divisor = int(round(1.0 / scale))

    if divisor <= 0:
        raise ValueError("scale 设置不合法。")

    h, w = image.shape[:2]

    target_h = ((h + divisor - 1) // divisor) * divisor
    target_w = ((w + divisor - 1) // divisor) * divisor

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h == 0 and pad_w == 0:
        return image, False

    padded = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    return padded, True


# =========================================================
# 8. 判断是否为图像文件
# =========================================================
def is_image_file(filename: str) -> bool:
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ext = os.path.splitext(filename)[1].lower()
    return ext in valid_exts


# =========================================================
# 9. 单张图像处理：固定退化 / 随机退化共用
# =========================================================
def data_resize(
        path: str,
        savepath: str,
        scale: float,
        blur_kernel: int = 3,
        blur_sigma: float = 0.8,
        overwrite_hr: bool = True,
        cv2interpolation: int = cv2.INTER_CUBIC,
        random_cfg: Optional[RandomDegradationConfig] = None,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = True
) -> bool:
    """
    单张图像生成 LR。
    参数
    ----
    path:
        HR 图像路径。
    savepath:
        LR 图像保存文件夹。
    scale:
        下采样比例。
        例如 scale = 0.25 表示 4× SR。
    blur_kernel:
        固定退化模式下的 Gaussian blur kernel。
    blur_sigma:
        固定退化模式下的 Gaussian blur sigma。
    overwrite_hr:
        如果 HR 图像尺寸不能被 scale 整除，padding 后是否覆盖原始 HR。
    cv2interpolation:
        固定退化模式下的下采样插值方法。
    random_cfg:
        随机退化配置。
        random_cfg.enabled=True 时，会随机采样 blur kernel、sigma 和 interpolation。
    rng:
        随机数生成器。
    verbose:
        是否打印日志。
    返回
    ----
    bool:
        是否成功处理。
    """
    if rng is None:
        rng = np.random.default_rng()

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"[ERROR] 读取失败：{path}")
        return False

    original_shape = image.shape
    # =========================
    # padding
    # =========================
    image, padded = pad_to_divisible(image, scale)
    if padded:
        if verbose:
            print(
                f"[PAD] {os.path.basename(path)} "
                f"{original_shape} -> {image.shape}"
            )

        if overwrite_hr:
            cv2.imwrite(
                path,
                image,
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
            if verbose:
                print(f"[OVERWRITE HR] {path}")
    # =========================
    # 采样退化参数  如果 random_cfg  中 设置为true 则 进行随机化的 操作 ，否则采用 固定的 退化 数值
    # =========================
    params = sample_degradation_params(
        fixed_blur_kernel=blur_kernel,
        fixed_blur_sigma=blur_sigma,
        fixed_interpolation=cv2interpolation,
        random_cfg=random_cfg,
        rng=rng
    )
    # =========================
    # Gaussian blur
    # =========================
    if params.blur_kernel > 1:
        image = cv2.GaussianBlur(
            image,
            (params.blur_kernel, params.blur_kernel),
            params.blur_sigma
        )

    # =========================
    # downsample
    # =========================
    lr = cv2.resize(
        image,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=params.interpolation
    )

    os.makedirs(savepath, exist_ok=True)

    name = os.path.basename(path)
    out_path = os.path.join(savepath, name)
    ok = cv2.imwrite(
        out_path,
        lr,
        [cv2.IMWRITE_PNG_COMPRESSION, 0]
    )
    if not ok:
        print(f"[ERROR] 保存失败：{out_path}")
        return False
    if verbose:
        mode = "RANDOM" if params.random_enabled else "FIXED"
        print(
            f"[LR SAVED] {name} "
            f"{image.shape} -> {lr.shape} | "
            f"mode={mode}, "
            f"kernel={params.blur_kernel}, "
            f"sigma={params.blur_sigma:.3f}, "
            f"interpolation={interpolation_name(params.interpolation)}"
        )
    return True


# =========================================================
# 10. 普通 batch 处理版本
# =========================================================
def batch_process(
        folder: str,
        savepath: str,
        scale: float,
        blur_kernel: int = 3,
        blur_sigma: float = 0.8,
        overwrite_hr: bool = True,
        cv2interpolation: int = cv2.INTER_CUBIC,
        random_cfg: Optional[RandomDegradationConfig] = None,
        verbose: bool = True
) -> None:
    """
    普通单线程 batch 处理版本。
    适合调试，日志清晰。
    """
    os.makedirs(savepath, exist_ok=True)

    image_list: List[str] = [
        name for name in os.listdir(folder)
        if is_image_file(name)
    ]

    image_list.sort()

    total = len(image_list)

    if total == 0:
        print(f"[WARNING] 文件夹中没有找到图像：{folder}")
        return

    rng = np.random.default_rng(
        None if random_cfg is None else random_cfg.seed
    )

    success_count = 0

    for idx, name in enumerate(image_list, start=1):
        image_path = os.path.join(folder, name)

        ok = data_resize(
            path=image_path,
            savepath=savepath,
            scale=scale,
            blur_kernel=blur_kernel,
            blur_sigma=blur_sigma,
            overwrite_hr=overwrite_hr,
            cv2interpolation=cv2interpolation,
            random_cfg=random_cfg,
            rng=rng,
            verbose=verbose
        )

        if ok:
            success_count += 1

        print(f"[PROGRESS] {idx}/{total}")

    print(f"[DONE] 成功处理 {success_count}/{total} 张图像。")


# =========================================================
# 11. 多线程加速 batch 版本
# =========================================================
def batch_process_fast(
        folder: str,
        savepath: str,
        scale: float,
        blur_kernel: int = 3,
        blur_sigma: float = 0.8,
        overwrite_hr: bool = True,
        cv2interpolation: int = cv2.INTER_CUBIC,
        random_cfg: Optional[RandomDegradationConfig] = None,
        num_workers: int = 8,
        verbose: bool = False
) -> None:
    """
    多线程加速 batch 处理版本。

    适合大量图像。
    OpenCV 的 imread / resize / GaussianBlur / imwrite 底层通常会释放 GIL，
    所以 ThreadPoolExecutor 对这类任务一般有加速效果。

    注意：
    1. 如果 overwrite_hr=True，多线程会同时写不同文件，一般没问题。
    2. 如果你的硬盘很慢，num_workers 不宜过大。
    3. 建议先用普通版测试无误，再切到 fast 版本。
    """
    os.makedirs(savepath, exist_ok=True)

    image_list: List[str] = [
        name for name in os.listdir(folder)
        if is_image_file(name)
    ]

    image_list.sort()

    total = len(image_list)

    if total == 0:
        print(f"[WARNING] 文件夹中没有找到图像：{folder}")
        return

    # 避免 OpenCV 自己开太多线程，再叠加 Python 多线程导致过度并发
    cv2.setNumThreads(1)

    base_seed: int = 0
    if random_cfg is not None and random_cfg.seed is not None:
        base_seed = int(random_cfg.seed)

    success_count = 0

    def worker(index: int, filename: str) -> bool:
        image_path = os.path.join(folder, filename)

        # 每张图独立随机种子，保证多线程下也可复现
        local_rng = np.random.default_rng(base_seed + index)

        return data_resize(
            path=image_path,
            savepath=savepath,
            scale=scale,
            blur_kernel=blur_kernel,
            blur_sigma=blur_sigma,
            overwrite_hr=overwrite_hr,
            cv2interpolation=cv2interpolation,
            random_cfg=random_cfg,
            rng=local_rng,
            verbose=verbose
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(worker, idx, name): name
            for idx, name in enumerate(image_list, start=1)
        }

        finished = 0

        for future in as_completed(futures):
            filename = futures[future]

            try:
                ok = future.result()
            except Exception as e:
                ok = False
                print(f"[ERROR] {filename} 处理异常：{e}")

            finished += 1

            if ok:
                success_count += 1

            if finished % 100 == 0 or finished == total:
                print(f"[PROGRESS] {finished}/{total}")

    print(f"[DONE] 成功处理 {success_count}/{total} 张图像。")


# =========================================================
# 12. main
# =========================================================
if __name__ == "__main__":

    folder = r"/home/liangshubo/Project/CTSR/dataset/rawdata/DeepLesion/test/HR_all"
    save_path = r"/home/liangshubo/Project/CTSR/dataset/rawdata/DeepLesion/test/HR_k1s0_areadown_x4"

    scale = 0.25  # DOWN 4× SR

    # =====================================================
    # 固定退化参数
    # 当 random_cfg.enabled=False 时，使用这里的固定参数。
    # =====================================================
    fixed_blur_kernel = 1
    fixed_blur_sigma = 0
    fixed_interpolation = cv2.INTER_AREA

    # =====================================================
    # 随机退化配置
    # enabled=True 开启随机退化
    # enabled=False 则退回到固定退化
    # =====================================================
    random_cfg = RandomDegradationConfig(
        enabled=False,
        blur_prob=0.9,
        blur_kernel_choices=(3, 5, 7),
        blur_sigma_range=(0.8, 1.4),
        interpolation_choices=(
            (cv2.INTER_AREA, 0.50),
            (cv2.INTER_LINEAR, 0.25),
            (cv2.INTER_CUBIC, 0.25),
        ),
        seed=2026
    )

    # =====================================================
    # 版本 1：普通版，适合调试
    # =====================================================
    batch_process(
        folder=folder,
        savepath=save_path,
        scale=scale,
        blur_kernel=fixed_blur_kernel,
        blur_sigma=fixed_blur_sigma,
        overwrite_hr=True,
        cv2interpolation=fixed_interpolation,
        random_cfg=random_cfg,
        verbose=True
    )

    # =====================================================
    # 版本 2：多线程加速版，适合大量图像
    # =====================================================
    # batch_process_fast(
    #     folder=folder,
    #     savepath=save_path,
    #     scale=scale,
    #     blur_kernel=fixed_blur_kernel,
    #     blur_sigma=fixed_blur_sigma,
    #     overwrite_hr=True,
    #     cv2interpolation=fixed_interpolation,
    #     random_cfg=random_cfg,
    #     num_workers=8,
    #     verbose=False
    # )