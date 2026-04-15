import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

# 定义评估函数（与步骤1一致）
def calc_metrics(gt, pred):
    gt_bin = gt > 0
    pred_bin = pred > 0
    iou = jaccard_score(gt_bin.flatten(), pred_bin.flatten(), average='binary', zero_division=0)
    pa = np.mean(gt_bin.flatten() == pred_bin.flatten())
    prec = precision_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    rec = recall_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    f1 = f1_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)
    return iou, pa, prec, rec, f1

# ------------------------------
# 配置路径
# ------------------------------
# 获取当前脚本所在目录（scripts/）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级到达 formal_project 根目录
project_root = os.path.dirname(os.path.dirname(script_dir))  # .../formal_project

# 基于项目根目录构建各路径
input_dir = os.path.join(project_root, "seg_compare", "test_image")
gt_dir = os.path.join(project_root, "seg_compare", "test_gt")
yolo_out_dir = os.path.join(project_root, "seg_compare", "pred_yolo")
model_path = os.path.join(project_root, "yolov8_model", "wheelchock5_best.pt")

os.makedirs(yolo_out_dir, exist_ok=True)

# ------------------------------
# 加载YOLO模型
# ------------------------------
model = YOLO(model_path)

# ------------------------------
# 批量处理并计时
# ------------------------------
yolo_times = []
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.png','.jpg','.jpeg')):
        continue
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    start = time.perf_counter()
    results = model(img, verbose=False)
    if results[0].masks is None:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    else:
        mask = results[0].masks.data[0].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        # 将掩码缩放到原图尺寸
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    elapsed = time.perf_counter() - start
    yolo_times.append(elapsed)
    out_path = os.path.join(yolo_out_dir, fname)
    cv2.imwrite(out_path, mask)
    print(f"YOLO: {fname} done, time={elapsed*1000:.2f}ms")

print(f"YOLO average time: {np.mean(yolo_times)*1000:.2f} ms")

# ------------------------------
# 评估YOLO方法
# ------------------------------
yolo_metrics = []
for fname in os.listdir(gt_dir):
    if not fname.lower().endswith('.png'):
        continue
    gt_path = os.path.join(gt_dir, fname)
    pred_path = os.path.join(yolo_out_dir, fname)
    if not os.path.exists(pred_path):
        continue
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if gt is None or pred is None:
        continue
    yolo_metrics.append(calc_metrics(gt, pred))

if yolo_metrics:
    yolo_mean = np.mean(yolo_metrics, axis=0)
    yolo_std = np.std(yolo_metrics, axis=0)
    print("\n=== YOLOv8-seg Evaluation ===")
    print(f"mIoU: {yolo_mean[0]:.4f} ± {yolo_std[0]:.4f}")
    print(f"Pixel Acc: {yolo_mean[1]:.4f} ± {yolo_std[1]:.4f}")
    print(f"Precision: {yolo_mean[2]:.4f} ± {yolo_std[2]:.4f}")
    print(f"Recall: {yolo_mean[3]:.4f} ± {yolo_std[3]:.4f}")
    print(f"F1: {yolo_mean[4]:.4f} ± {yolo_std[4]:.4f}")