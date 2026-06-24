import cv2

# 全局变量
drawing = False          # 是否正在绘制
start_point = (-1, -1)   # 起始点 (x, y)
end_point = (-1, -1)     # 当前终点 (x, y)
img_original = None      # 原始图像副本
img_display = None       # 用于显示的图像（带临时矩形）

def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, drawing, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键按下，开始绘制
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
        img_display = img_original.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 实时更新终点，并重绘图像+矩形
            end_point = (x, y)
            img_display = img_original.copy()
            cv2.rectangle(img_display, start_point, end_point, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        # 左键松开，结束绘制
        drawing = False
        end_point = (x, y)
        img_display = img_original.copy()
        cv2.rectangle(img_display, start_point, end_point, (0, 0, 255), 2)

        # 确保左上和右下（处理反向拖拽）
        x1, y1 = start_point
        x2, y2 = end_point
        left_top = (min(x1, x2), min(y1, y2))
        right_bottom = (max(x1, x2), max(y1, y2))

        # 输出格式为 (H, W)，即 (y, x)
        print(f"左上角 (H1, W1): ({left_top[1]}, {left_top[0]})")
        print(f"右下角 (H2, W2): ({right_bottom[1]}, {right_bottom[0]})")
        print("-" * 30)

def main(image_path):
    global img_original, img_display

    # 读取图像
    img_original = cv2.imread(image_path)
    if img_original is None:
        print("❌ 无法加载图像，请检查路径是否正确。")
        return

    img_display = img_original.copy()

    # 创建窗口并设置鼠标回调
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    print("🖱️  在图像上按住鼠标左键拖动以选择区域，松开后输出坐标。")
    print("按 'q' 或关闭窗口退出。")

    while True:
        cv2.imshow("Image", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 显式定义图像路径
    image_path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/N6000_1126/png/train/image/20251121-1_20251121154827_2_1.png"  # 替换这里的路径为你实际使用的图像路径
    main(image_path)