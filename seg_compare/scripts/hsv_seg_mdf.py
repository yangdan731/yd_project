import os
import cv2
import numpy as np
import time
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

# ------------------------------
# 颜色阈值分割函数
# ------------------------------
def segment_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 取最大轮廓区域（可选）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            pad = 10
            x = max(0, x-pad)
            y = max(0, y-pad)
            w = min(image.shape[1]-x, w+2*pad)
            h = min(image.shape[0]-y, h+2*pad)
            roi_mask = np.zeros_like(mask)
            roi_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
            return roi_mask
    return mask

# ------------------------------
# 批量处理颜色方法
# ------------------------------
# 获取当前脚本所在目录的父目录（即 seg_compare 目录）
script_dir = os.path.dirname(os.path.abspath(__file__))      # scripts/
base_dir = os.path.dirname(script_dir)                      # seg_compare/

# 构建路径
input_dir = os.path.join(base_dir, "test_image")
gt_dir = os.path.join(base_dir, "test_gt")
color_out_dir = os.path.join(base_dir, "pred_color")
os.makedirs(color_out_dir, exist_ok=True)

color_times = []
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.png','.jpg','.jpeg')):
        continue
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    start = time.perf_counter()
    mask = segment_color(img)
    elapsed = time.perf_counter() - start
    color_times.append(elapsed)
    out_path = os.path.join(color_out_dir, fname)
    cv2.imwrite(out_path, mask)
    print(f"Color method: {fname} done, time={elapsed*1000:.2f}ms")

print(f"Color method avg time: {np.mean(color_times)*1000:.2f} ms")

# ------------------------------
# 评估颜色方法
# ------------------------------
def calc_metrics(gt, pred):
    gt_bin = gt > 0
    pred_bin = pred > 0
    iou = jaccard_score(gt_bin.flatten(), pred_bin.flatten(), average='binary', zero_division=0)
    pa = np.mean(gt_bin.flatten() == pred_bin.flatten())
    prec = precision_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    rec = recall_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    f1 = f1_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    return iou, pa, prec, rec, f1

color_metrics = []
for fname in os.listdir(gt_dir):
    if not fname.lower().endswith('.png'):
        continue
    gt_path = os.path.join(gt_dir, fname)
    pred_path = os.path.join(color_out_dir, fname)
    if not os.path.exists(pred_path):
        continue
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if gt is None or pred is None:
        continue
    color_metrics.append(calc_metrics(gt, pred))

if color_metrics:
    color_mean = np.mean(color_metrics, axis=0)
    color_std = np.std(color_metrics, axis=0)
    print("\n=== Color Threshold Method ===")
    print(f"mIoU: {color_mean[0]:.4f} ± {color_std[0]:.4f}")
    print(f"Pixel Acc: {color_mean[1]:.4f} ± {color_std[1]:.4f}")
    print(f"Precision: {color_mean[2]:.4f} ± {color_std[2]:.4f}")
    print(f"Recall: {color_mean[3]:.4f} ± {color_std[3]:.4f}")
    print(f"F1: {color_mean[4]:.4f} ± {color_std[4]:.4f}")