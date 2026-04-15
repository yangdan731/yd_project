import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 配置参数
MODEL_PATH = r"yolov8_model/wheelchock5_best.pt"
RGB_DIR = r"extracted_data2/rgb"          # 存放多帧 RGB 图像的目录
OUTPUT_DIR = r"visualization/multi_vis2"  # 输出目录

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
model = YOLO(MODEL_PATH)

# 获取所有 RGB 图像文件（按文件名排序）
rgb_files = sorted([f for f in os.listdir(RGB_DIR) if f.endswith('.png')])
print(f"找到 {len(rgb_files)} 张图像")

for i, filename in enumerate(rgb_files):
    rgb_path = os.path.join(RGB_DIR, filename)
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f"无法读取图像: {rgb_path}")
        continue

    # YOLOv8-seg 推理
    results = model(rgb)

    if results[0].masks is None:
        print(f"{filename}: 未检测到目标，跳过")
        continue

    # 取第一个检测到的目标的掩码
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)

    # 将 mask 缩放到原始图像尺寸（因为模型推理时可能自动缩放了）
    mask_original = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_original = (mask_original > 0.5).astype(np.uint8)

    # 生成 RGB + 掩码叠加图
    overlay = rgb.copy()
    color_mask = np.zeros_like(rgb)
    color_mask[:, :, 2] = (mask_original * 255).astype(np.uint8)   # 红色通道
    overlay = cv2.addWeighted(rgb, 0.6, color_mask, 0.4, 0)

    # 绘制掩码轮廓（绿色）
    contours, _ = cv2.findContours(mask_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # 保存结果
    out_path = os.path.join(OUTPUT_DIR, filename)  # 同名保存
    cv2.imwrite(out_path, overlay)
    print(f"已保存: {out_path}")

print("批量可视化完成！")