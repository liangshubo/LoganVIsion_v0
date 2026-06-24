def _draw_contours_with_ordered_legend(overlay_image, pred_mask, min_area=500, thickness=2,
                                       draw_only_max_box=False, background_class=0):
    """
    1. 在 overlay_image 上绘制预测 mask 的轮廓。
    2. 对同类别的多个实例按面积从大到小排序并编号 (如 Nerve_1, Nerve_2)。
    3. 图片左上角显示详细信息列表 (Nerve_1 : Area 1000 px)。
    4. Mask 轮廓左上角仅标注编号名称 (Nerve_1)。
    """
    piex2mm = 0.05

    img_out = overlay_image.copy()

    if img_out.ndim == 2:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
        # print(img_out.shape)

    # 颜色表 & 类名
    VOC_COLORMAP = [(0, 0, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0), (128, 128, 0), (255, 128, 128)]
    CLASS_NAMES = {0: "Background",
                   1: "Nerve",
                   2: "Muscle",
                   3: "Artery",
                   4: "Vein"}

    unique_labels = sorted([int(l) for l in np.unique(pred_mask) if int(l) != background_class])

    if not unique_labels:
        return img_out

    # 用于存储后续需要绘制的文字任务，避免频繁转换 PIL/OpenCV
    # 格式: {'text': str, 'pos': (x, y), 'color': (r, g, b), 'type': 'local'/'legend'}
    text_tasks = []

    # 左上角列表的起始 Y 坐标
    legend_start_y = 10
    legend_line_height = 25  # 每行文字的高度

    for label in unique_labels:
        # 1. 提取 Mask 并找轮廓
        bin_mask = (pred_mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # 2. 筛选面积并排序 (关键步骤：按面积从大到小排序)
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                valid_contours.append((cnt, area))

        # 按面积降序排列: list of (cnt, area)
        valid_contours.sort(key=lambda x: x[1], reverse=True)

        # 如果只画最大的，就切片取第一个
        if draw_only_max_box and valid_contours:
            valid_contours = valid_contours[:1]

        if not valid_contours:
            continue

        # 获取颜色
        color_bgr = VOC_COLORMAP[label] if label < len(VOC_COLORMAP) else [128, 128, 128]

        # print(color_bgr)
        color_rgb = tuple(color_bgr[::-1])  # PIL 需要 RGB

        # 3. 遍历该类的所有实例
        for idx, (cnt, area) in enumerate(valid_contours):
            instance_id = idx + 1  # 1, 2, 3...
            class_name = CLASS_NAMES.get(label, f"Class{label}")
            instance_name = f"{class_name}_{instance_id}"  # 例如 Nerve_1

            # --- OpenCV 绘制轮廓 ---
            cv2.drawContours(img_out, [cnt], -1, color=color_bgr, thickness=thickness)

            # --- 收集文字任务：局部标签 (轮廓旁) ---
            x, y, w, h = cv2.boundingRect(cnt)
            text_tasks.append({
                'type': 'local',
                'text': instance_name,
                'pos': (x, max(0, y - 20)),  # 放在轮廓上方
                'color': color_rgb,  # 局部标签用白色字
            })

            # --- 收集文字任务：全局列表 (左上角) ---
            legend_text = f"{instance_name} : Area {int(area) * piex2mm * piex2mm:.4f } mm²"
            text_tasks.append({
                'type': 'legend',
                'text': legend_text,
                'pos': (10, legend_start_y),
                'color': color_rgb,  # 列表文字颜色与Mask颜色一致，方便对应
            })

            # 更新列表的 Y 坐标，为下一行腾出空间
            legend_start_y += legend_line_height

    # --- 4. 统一转 PIL 绘制所有文字 ---
    pil_img = Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 字体设置
    try:
        font_path = "/home/ubuntu4090/下载/HarmonyOS Sans /HarmonyOS_Sans_SC/HarmonyOS_SansSC_Medium.ttf"
        font_local = ImageFont.truetype(font_path, 14)  # 局部标签字体
        font_legend = ImageFont.truetype(font_path, 16)  # 列表字体稍大
    except IOError:
        font_local = ImageFont.load_default()
        font_legend = ImageFont.load_default()

    # 为了防止左上角列表看不清，可以先在左上角画一个半透明黑底背景 (可选)
    # 计算列表区域的总高度
    # if legend_start_y > 10:
    #    draw.rectangle([(5, 5), (250, legend_start_y + 5)], fill=(0, 0, 0, 120))  # RGBA, A=120半透明

    for task in text_tasks:
        text = task['text']
        x, y = task['pos']
        color = task['color']

        if task['type'] == 'local':
            # 绘制局部标签 (带背景框)
            # 获取文字大小
            try:
                bbox = draw.textbbox((0, 0), text, font=font_local)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                w, h = draw.textsize(text, font=font_local)

            # 画文字背景框
            # draw.rectangle([(x, y), (x + w + 4, y + h + 4)], fill=(0, 0, 0))
            # 画文字
            draw.text((x + 2, y + 2), text, font=font_local, fill=color)

        elif task['type'] == 'legend':
            # 绘制左上角列表 (无背景框，或者上面已经统一画了大背景)
            # 这里让字体颜色 = 类别颜色，方便区分
            draw.text((x, y), text, font=font_legend, fill=color)

    # 转回 OpenCV
    img_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img_out